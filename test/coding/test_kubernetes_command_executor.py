from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autogen.coding.base import CodeBlock
from autogen.coding.kubernetes_commandline_code_executor import KubernetesCommandLineCodeExecutor

# Begin test cases


@patch("kubernetes.client.CoreV1Api")
@patch("kubernetes.config.load_kube_config")
def test_executor_initialization(mock_load_kube_config, mock_core_v1_api):
    # Mock Kubernetes configuration loading
    mock_load_kube_config.return_value = None

    # Mock the CoreV1Api client
    mock_core_v1_api_instance = MagicMock()
    mock_core_v1_api.return_value = mock_core_v1_api_instance

    # Mock read_namespaced_pod to return a pod with status.phase == "Running"
    pod_mock = MagicMock()
    pod_mock.status.phase = "Running"
    mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

    # Initialize the executor
    executor = KubernetesCommandLineCodeExecutor(
        image="python:3-slim", namespace="default", timeout=60, pod_name="test-pod"
    )

    # Assertions
    assert executor.image == "python:3-slim"
    assert executor.namespace == "default"
    assert executor._timeout == 60
    assert executor.pod_name == "test-pod"  # Since we provided a name
    assert executor._work_dir == Path(".")


def test_execute_code_blocks():
    # Mock internal methods
    with patch.object(KubernetesCommandLineCodeExecutor, "_upload_file_to_pod") as mock_upload, patch.object(
        KubernetesCommandLineCodeExecutor, "_exec_command_with_exit_code"
    ) as mock_exec, patch("kubernetes.client.CoreV1Api") as mock_core_v1_api:

        # Setup mocks
        mock_upload.return_value = None
        mock_exec.return_value = (0, "Hello, Kubernetes Executor!\n")

        # Mock Kubernetes API client
        mock_core_v1_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_core_v1_api_instance

        # Mock read_namespaced_pod to return a pod with status.phase == "Running"
        pod_mock = MagicMock()
        pod_mock.status.phase = "Running"
        mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

        # Create code blocks
        code_blocks = [CodeBlock(language="python", code="print('Hello, Kubernetes Executor!')")]

        # Initialize the executor
        executor = KubernetesCommandLineCodeExecutor()

        # Execute the code blocks
        result = executor.execute_code_blocks(code_blocks)

        # Assertions
        assert result.exit_code == 0
        assert result.output == "Hello, Kubernetes Executor!\n"
        assert result.code_file.startswith("/workspace/tmp_code_")
        assert result.code_file.endswith(".python")

        # Ensure the upload and exec methods were called
        mock_upload.assert_called_once()
        mock_exec.assert_called_once()


def test_execute_code_blocks_with_error():
    # Mock internal methods
    with patch.object(KubernetesCommandLineCodeExecutor, "_upload_file_to_pod") as mock_upload, patch.object(
        KubernetesCommandLineCodeExecutor, "_exec_command_with_exit_code"
    ) as mock_exec, patch("kubernetes.client.CoreV1Api") as mock_core_v1_api:

        # Setup mocks
        mock_upload.return_value = None
        mock_exec.return_value = (1, "SyntaxError: invalid syntax\n")

        # Mock Kubernetes API client
        mock_core_v1_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_core_v1_api_instance

        # Mock read_namespaced_pod to return a pod with status.phase == "Running"
        pod_mock = MagicMock()
        pod_mock.status.phase = "Running"
        mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

        # Create code blocks with a syntax error
        code_blocks = [CodeBlock(language="python", code="print('Hello, World!'\n")]

        # Initialize the executor
        executor = KubernetesCommandLineCodeExecutor()

        # Execute the code blocks
        result = executor.execute_code_blocks(code_blocks)

        # Assertions
        assert result.exit_code == 1
        assert result.output == "SyntaxError: invalid syntax\n"
        assert result.code_file.startswith("/workspace/tmp_code_")
        assert result.code_file.endswith(".python")

        # Ensure the upload and exec methods were called
        mock_upload.assert_called_once()
        mock_exec.assert_called_once()


def test_unsupported_language():
    with patch("kubernetes.client.CoreV1Api") as mock_core_v1_api:
        # Mock Kubernetes API client
        mock_core_v1_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_core_v1_api_instance

        # Mock read_namespaced_pod to return a pod with status.phase == "Running"
        pod_mock = MagicMock()
        pod_mock.status.phase = "Running"
        mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

        # Initialize the executor
        executor = KubernetesCommandLineCodeExecutor()

        # Create code blocks with an unsupported language
        code_blocks = [CodeBlock(language="ruby", code="puts 'Hello, World!'")]

        # Execute the code blocks
        result = executor.execute_code_blocks(code_blocks)

        # Assertions
        assert result.exit_code == 1
        assert "Unsupported language ruby" in result.output


def test_no_code_blocks():
    with patch("kubernetes.client.CoreV1Api") as mock_core_v1_api:
        # Mock Kubernetes API client
        mock_core_v1_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_core_v1_api_instance

        # Mock read_namespaced_pod to return a pod with status.phase == "Running"
        pod_mock = MagicMock()
        pod_mock.status.phase = "Running"
        mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

        # Initialize the executor
        executor = KubernetesCommandLineCodeExecutor()

        # Execute with no code blocks
        with pytest.raises(ValueError) as excinfo:
            executor.execute_code_blocks([])

        # Assertions
        assert "No code blocks to execute." in str(excinfo.value)


def test_timeout_handling():
    # Mock internal methods
    with patch.object(KubernetesCommandLineCodeExecutor, "_upload_file_to_pod") as mock_upload, patch.object(
        KubernetesCommandLineCodeExecutor, "_exec_command_with_exit_code"
    ) as mock_exec, patch("kubernetes.client.CoreV1Api") as mock_core_v1_api:

        # Setup mocks to simulate a timeout exit code
        mock_upload.return_value = None
        mock_exec.return_value = (124, "Timeout occurred\n")

        # Mock Kubernetes API client
        mock_core_v1_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_core_v1_api_instance

        # Mock read_namespaced_pod to return a pod with status.phase == "Running"
        pod_mock = MagicMock()
        pod_mock.status.phase = "Running"
        mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

        # Create code blocks
        code_blocks = [CodeBlock(language="python", code="while True: pass")]

        # Initialize the executor with a short timeout
        executor = KubernetesCommandLineCodeExecutor(timeout=1)

        # Execute the code blocks
        result = executor.execute_code_blocks(code_blocks)

        # Assertions
        assert result.exit_code == 124  # Exit code for timeout in Unix
        assert "Timeout occurred" in result.output


def test_restart_executor():
    # Mock internal methods
    with patch.object(KubernetesCommandLineCodeExecutor, "stop") as mock_stop, patch.object(
        KubernetesCommandLineCodeExecutor, "_create_pod"
    ) as mock_create_pod, patch("kubernetes.client.CoreV1Api") as mock_core_v1_api:

        # Mock Kubernetes API client
        mock_core_v1_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_core_v1_api_instance

        # Mock read_namespaced_pod to return a pod with status.phase == "Running"
        pod_mock = MagicMock()
        pod_mock.status.phase = "Running"
        mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

        # Initialize the executor
        executor = KubernetesCommandLineCodeExecutor()

        # Reset the mock to zero calls after initialization
        mock_create_pod.reset_mock()
        mock_stop.reset_mock()

        # Call restart
        executor.restart()

        # Assertions
        mock_stop.assert_called_once()
        mock_create_pod.assert_called_once()


def test_executor_context_manager():
    # Mock the stop method
    with patch.object(KubernetesCommandLineCodeExecutor, "stop") as mock_stop, patch(
        "kubernetes.client.CoreV1Api"
    ) as mock_core_v1_api:

        # Mock Kubernetes API client
        mock_core_v1_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_core_v1_api_instance

        # Mock read_namespaced_pod to return a pod with status.phase == "Running"
        pod_mock = MagicMock()
        pod_mock.status.phase = "Running"
        mock_core_v1_api_instance.read_namespaced_pod.return_value = pod_mock

        # Use the executor as a context manager
        with KubernetesCommandLineCodeExecutor() as executor:
            assert executor is not None

        # Ensure stop was called upon exit
        mock_stop.assert_called_once()
