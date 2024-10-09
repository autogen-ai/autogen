from __future__ import annotations

import atexit
import shlex
import sys
import time
import uuid
from hashlib import md5
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

from kubernetes import client, config, stream
from kubernetes.client.exceptions import ApiException

from ..code_utils import TIMEOUT_MSG, _cmd
from .base import CodeBlock, CodeExecutor, CodeExtractor, CommandLineCodeResult
from .markdown_code_extractor import MarkdownCodeExtractor
from .utils import _get_file_name_from_content, silence_pip

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ("KubernetesCommandLineCodeExecutor",)


class KubernetesCommandLineCodeExecutor(CodeExecutor):
    DEFAULT_EXECUTION_POLICY: ClassVar[Dict[str, bool]] = {
        "bash": True,
        "shell": True,
        "sh": True,
        "pwsh": True,
        "powershell": True,
        "ps1": True,
        "python": True,
        "javascript": False,
        "html": False,
        "css": False,
    }
    LANGUAGE_ALIASES: ClassVar[Dict[str, str]] = {"py": "python", "js": "javascript"}

    def __init__(
        self,
        image: str = "python:3-slim",
        pod_name: Optional[str] = None,
        namespace: str = "default",
        timeout: int = 60,
        work_dir: Union[Path, str] = Path("."),
        execution_policies: Optional[Dict[str, bool]] = None,
    ):
        """(Experimental) A code executor class that executes code through
        a command line environment in a Kubernetes pod.

        Args:
            image (str, optional): Docker image to use for code execution.
                Defaults to "python:3-slim".
            pod_name (Optional[str], optional): Name of the Kubernetes pod.
                If None, will autogenerate a name. Defaults to None.
            namespace (str, optional): Kubernetes namespace to create the pod in.
                Defaults to "default".
            timeout (int, optional): The timeout for code execution. Defaults to 60.
            work_dir (Union[Path, str], optional): The working directory path in the pod.
                Defaults to Path(".").
            execution_policies (Optional[Dict[str, bool]], optional): Execution policies
                for different languages. Defaults to None.

        Raises:
            ValueError: On argument error, or if the pod fails to start.
        """
        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)
        self._work_dir: Path = work_dir

        self.image = image
        self.namespace = namespace
        self._timeout = timeout
        self.execution_policies = self.DEFAULT_EXECUTION_POLICY.copy()
        if execution_policies is not None:
            self.execution_policies.update(execution_policies)

        if pod_name is None:
            pod_name = f"autogen-code-exec-{uuid.uuid4()}"
        self.pod_name = pod_name

        # Load Kubernetes configuration
        try:
            config.load_kube_config()
        except config.ConfigException:
            config.load_incluster_config()

        # Create Kubernetes API client
        self.api_instance = client.CoreV1Api()

        # Create the pod
        self._create_pod()

        def cleanup() -> None:
            self.stop()
            atexit.unregister(cleanup)

        atexit.register(cleanup)
        self._cleanup = cleanup

    def _create_pod(self) -> None:
        # Define the pod spec
        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(name=self.pod_name),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name="executor",
                        image=self.image,
                        command=["/bin/sh"],
                        tty=True,
                        stdin=True,
                        volume_mounts=[client.V1VolumeMount(mount_path="/workspace", name="workspace-volume")],
                    )
                ],
                volumes=[client.V1Volume(name="workspace-volume", empty_dir=client.V1EmptyDirVolumeSource())],
                restart_policy="Never",
            ),
        )

        # Create the pod
        try:
            self.api_instance.create_namespaced_pod(namespace=self.namespace, body=pod)
        except ApiException as e:
            raise ValueError(f"Failed to create pod: {e}")

        # Wait for the pod to be running
        self._wait_for_pod_ready(self._timeout)

    def _wait_for_pod_ready(self, timeout: int = 60) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                pod = self.api_instance.read_namespaced_pod(name=self.pod_name, namespace=self.namespace)
                if pod.status.phase == "Running":
                    return
                elif pod.status.phase in ["Failed", "Unknown", "Succeeded"]:
                    raise ValueError(f"Pod failed to start. Status: {pod.status.phase}")
            except ApiException as e:
                if e.status != 404:
                    raise e
            time.sleep(1)
        raise TimeoutError(f"Timed out waiting for pod {self.pod_name} to be running.")

    @property
    def timeout(self) -> int:
        """(Experimental) The timeout for code execution."""
        return self._timeout

    @property
    def work_dir(self) -> Path:
        """(Experimental) The working directory for the code execution."""
        return self._work_dir

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) Export a code extractor that can be used by an agent."""
        return MarkdownCodeExtractor()

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        """(Experimental) Execute the code blocks and return the result.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CommandLineCodeResult: The result of the code execution."""
        if len(code_blocks) == 0:
            raise ValueError("No code blocks to execute.")

        outputs = []
        last_exit_code = 0
        files = []
        for code_block in code_blocks:
            lang = self.LANGUAGE_ALIASES.get(code_block.language.lower(), code_block.language.lower())
            if lang not in self.DEFAULT_EXECUTION_POLICY:
                outputs.append(f"Unsupported language {lang}\n")
                last_exit_code = 1
                break

            execute_code = self.execution_policies.get(lang, False)
            code = silence_pip(code_block.code, lang)

            # Determine filename
            try:
                filename = _get_file_name_from_content(code, Path("/workspace"))
            except ValueError:
                outputs.append("Filename is not in the workspace")
                last_exit_code = 1
                break

            if not filename:
                filename = f"tmp_code_{md5(code.encode()).hexdigest()}.{lang}"
            code_path = f"/workspace/{filename}"
            files.append(code_path)

            # Upload the code to the pod
            self._upload_file_to_pod(code, code_path)

            if not execute_code:
                outputs.append(f"Code saved to {code_path}\n")
                continue

            # Execute the code
            exec_command = [_cmd(lang), code_path]
            exit_code, output = self._exec_command_with_exit_code(exec_command)
            outputs.append(output)

            last_exit_code = exit_code
            if exit_code != 0:
                break

        code_file = files[0] if files else None
        return CommandLineCodeResult(exit_code=last_exit_code, output="".join(outputs), code_file=code_file)

    def _upload_file_to_pod(self, content: str, remote_path: str) -> None:
        exec_command = ["sh", "-c", f"cat > {shlex.quote(remote_path)}"]
        resp = stream.stream(
            self.api_instance.connect_get_namespaced_pod_exec,
            name=self.pod_name,
            namespace=self.namespace,
            command=exec_command,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        try:
            resp.write_stdin(content)
            resp.update(timeout=5)
        except Exception as e:
            resp.close()
            raise e
        resp.close()

    def _exec_command_with_exit_code(self, command: List[str]) -> Tuple[int, str]:
        cmd_str = " ".join(shlex.quote(arg) for arg in command)
        # Wrap the command to capture the exit code
        wrapped_command = ["sh", "-c", f"{cmd_str}; echo $?"]
        try:
            resp = stream.stream(
                self.api_instance.connect_get_namespaced_pod_exec,
                name=self.pod_name,
                namespace=self.namespace,
                command=wrapped_command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
                _request_timeout=self._timeout,  # Set the timeout here
            )
            output = ""
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    output += resp.read_stdout()
                if resp.peek_stderr():
                    output += resp.read_stderr()
            resp.close()
        except Exception as e:
            # Handle timeout exception
            if isinstance(e, ApiException) and e.status == 504:
                output += "\n" + TIMEOUT_MSG
                exit_code = 124  # Similar to Unix timeout exit code
            else:
                raise e
        else:
            # Parse the exit code from the output
            output_lines = output.strip().splitlines()
            if output_lines:
                exit_code_str = output_lines[-1]
                try:
                    exit_code = int(exit_code_str)
                    output = "\n".join(output_lines[:-1])
                except ValueError:
                    # Could not parse exit code
                    exit_code = 1
            else:
                exit_code = 1
        return exit_code, output

    def restart(self) -> None:
        """(Experimental) Restart the code executor."""
        self.stop()
        # Recreate the pod
        self._create_pod()

    def stop(self) -> None:
        """(Experimental) Stop the code executor."""
        try:
            self.api_instance.delete_namespaced_pod(name=self.pod_name, namespace=self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise e

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        self.stop()
