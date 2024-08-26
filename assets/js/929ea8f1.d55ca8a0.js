"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8125],{89821:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>s,default:()=>g,frontMatter:()=>o,metadata:()=>r,toc:()=>l});var i=t(85893),a=t(11151);const o={custom_edit_url:"https://github.com/autogen-ai/autogen/edit/main/notebook/agentchat_agentoptimizer.ipynb",description:"AgentOptimizer is able to prompt LLMs to iteratively optimize function/skills of AutoGen agents according to the historical conversation and performance.",source_notebook:"/notebook/agentchat_agentoptimizer.ipynb",tags:["optimization","tool/function"],title:"AgentOptimizer: An Agentic Way to Train Your LLM Agent"},s="AgentOptimizer: An Agentic Way to Train Your LLM Agent",r={id:"notebooks/agentchat_agentoptimizer",title:"AgentOptimizer: An Agentic Way to Train Your LLM Agent",description:"AgentOptimizer is able to prompt LLMs to iteratively optimize function/skills of AutoGen agents according to the historical conversation and performance.",source:"@site/docs/notebooks/agentchat_agentoptimizer.mdx",sourceDirName:"notebooks",slug:"/notebooks/agentchat_agentoptimizer",permalink:"/autogen/docs/notebooks/agentchat_agentoptimizer",draft:!1,unlisted:!1,editUrl:"https://github.com/autogen-ai/autogen/edit/main/notebook/agentchat_agentoptimizer.ipynb",tags:[{label:"optimization",permalink:"/autogen/docs/tags/optimization"},{label:"tool/function",permalink:"/autogen/docs/tags/tool-function"}],version:"current",frontMatter:{custom_edit_url:"https://github.com/autogen-ai/autogen/edit/main/notebook/agentchat_agentoptimizer.ipynb",description:"AgentOptimizer is able to prompt LLMs to iteratively optimize function/skills of AutoGen agents according to the historical conversation and performance.",source_notebook:"/notebook/agentchat_agentoptimizer.ipynb",tags:["optimization","tool/function"],title:"AgentOptimizer: An Agentic Way to Train Your LLM Agent"},sidebar:"notebooksSidebar",previous:{title:"Agent Tracking with AgentOps",permalink:"/autogen/docs/notebooks/agentchat_agentops"},next:{title:"Task Solving with Code Generation, Execution and Debugging",permalink:"/autogen/docs/notebooks/agentchat_auto_feedback_from_code_execution"}},c={},l=[];function u(e){const n={a:"a",code:"code",h1:"h1",img:"img",p:"p",pre:"pre",...(0,a.a)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.h1,{id:"agentoptimizer-an-agentic-way-to-train-your-llm-agent",children:"AgentOptimizer: An Agentic Way to Train Your LLM Agent"}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.a,{href:"https://colab.research.google.com/github/autogen-ai/autogen/blob/main/notebook/agentchat_agentoptimizer.ipynb",children:(0,i.jsx)(n.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,i.jsx)(n.a,{href:"https://github.com/autogen-ai/autogen/blob/main/notebook/agentchat_agentoptimizer.ipynb",children:(0,i.jsx)(n.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,i.jsxs)(n.p,{children:["AutoGen offers conversable agents powered by LLM, tool, or human, which\ncan be used to perform tasks collectively via automated chat. This\nframework allows tool use and human participation through multi-agent\nconversation. Please find documentation about this feature\n",(0,i.jsx)(n.a,{href:"https://autogen-ai.github.io/autogen/docs/Use-Cases/agent_chat",children:"here"}),"."]}),"\n",(0,i.jsxs)(n.p,{children:["In traditional ML pipeline, we train a model by updating its parameter\naccording to the loss on the training set, while in the era of LLM\nagents, how should we train an agent? Here, we take an initial step\ntowards the agent training. Inspired by the ",(0,i.jsx)(n.a,{href:"https://platform.openai.com/docs/guides/function-calling",children:"function\ncalling"}),"\ncapabilities provided by OpenAI, we draw an analogy between model\nparameters and agent functions/skills, and update agent\u2019s\nfunctions/skills based on its historical performance on the training\nset. As an agentic way of training an agent, our approach help enhance\nthe agents\u2019 abilities without requiring access to the LLMs parameters."]}),"\n",(0,i.jsxs)(n.p,{children:["In this notebook, we introduce a new class, \u2018AgentOptimizer\u2019, which is\nable to improve the function list of one Assistant-UserProxy pair\naccording to the historical conversation histories. This feature would\nsupport agents in improving their ability to solve problems of the same\ntype as previous tasks. Specifically, given a set of training data,\nAgentOptimizer would iteratively prompt the LLM to optimize the existing\nfunction list of the AssistantAgent and UserProxyAgent with code\nimplementation if necessary. It also includes two strategies, roll-back,\nand early-stop, to streamline the training process. In the example\nscenario, we test the proposed AgentOptimizer in solving problems from\nthe ",(0,i.jsx)(n.a,{href:"https://github.com/hendrycks/math",children:"MATH dataset"}),"."]}),"\n",(0,i.jsxs)("figure",{children:[(0,i.jsx)("img",{src:"https://media.githubusercontent.com/media/autogen-ai/autogen/main/website/blog/2023-12-23-AgentOptimizer/img/agentoptimizer.png",alt:"AgentOptimizer"}),(0,i.jsx)("figcaption",{"aria-hidden":"true",children:"AgentOptimizer"})]}),"\n",(0,i.jsxs)(n.p,{children:["More information could be found in the\n",(0,i.jsx)(n.a,{href:"https://arxiv.org/abs/2402.11359",children:"paper"}),"."]}),"\n",(0,i.jsxs)(n.p,{children:["Authors: - ",(0,i.jsx)(n.a,{href:"https://github.com/skzhang1",children:"Shaokun Zhang"}),", Ph.D.\xa0student\nat the The Pennsylvania State University - ",(0,i.jsx)(n.a,{href:"https://jieyuz2.github.io",children:"Jieyu\nZhang"}),", Ph.D.\xa0student at the University of\nWashington"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import copy\nimport json\nimport os\nfrom typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union\n\nfrom openai import BadRequestError\n\nimport autogen\nfrom autogen import config_list_from_json\nfrom autogen.agentchat import Agent\nfrom autogen.agentchat.contrib.agent_optimizer import AgentOptimizer\nfrom autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent\nfrom autogen.code_utils import extract_code\nfrom autogen.math_utils import get_answer\n"})}),"\n",(0,i.jsx)(n.h1,{id:"mathuserproxy-with-function_call",children:"MathUserProxy with function_call"}),"\n",(0,i.jsxs)(n.p,{children:["This agent is a customized MathUserProxy inherits from its ",(0,i.jsx)(n.a,{href:"https://github.com/microsoft/autogen/blob/main/autogen/agentchat/contrib/math_user_proxy_agent.py",children:"parent\nclass"}),"."]}),"\n",(0,i.jsx)(n.p,{children:"It supports using both function_call and python to solve math problems."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'def is_termination_msg_mathchat(message):\n    """Check if a message is a termination message."""\n    if isinstance(message, dict):\n        message = message.get("content")\n        if message is None:\n            return False\n    cb = extract_code(message)\n    contain_code = False\n    for c in cb:\n        if c[0] == "python":\n            contain_code = True\n            break\n    if message.rstrip().find("TERMINATE") >= 0:\n        return True\n    return not contain_code and get_answer(message) is not None and get_answer(message) != ""\n\n\nclass MathUserProxyAgent(MathUserProxyAgent):\n    MAX_CONSECUTIVE_AUTO_REPLY = 15\n    DEFAULT_REPLY = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\\\boxed{}.)"\n    PROMPTS = """Let\'s solve a math problem.\nQuery requirements:\nYou should always use the \'print\' function for the output and use fractions/radical forms instead of decimals.\nYou can use packages like sympy to help you.\nYou must follow the formats below to write your code:\n```python\n# your code\n```\nIf some packages are missing, you could also suggest a code to install the corresponding package.\n\nPlease follow this process:\n1. Solve the problem step by step (do not over-divide the steps).\n2. Take out any queries that can be asked through Python code (for example, any calculations or equations that can be calculated) and functions you know in the context of this conversation.\n\nPlease\n(1) do not mix suggested Python codes and function calls in one step.\n(2) You MUST remember that you don\u2019t have a function named "python" available.\n\nYou must follow the formats below to write your Python code:\n```python\n# your code\n```\n\n3. Wait for me to give the results or wait for the executed results of the function call.\n4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.\n\nAfter all the queries are run and you get the answer, put the answer in \\\\boxed{}.\n\nProblem:\n"""\n\n    def __init__(\n        self,\n        name: Optional[str] = "MathChatAgent",\n        is_termination_msg: Optional[Callable[[Dict], bool]] = is_termination_msg_mathchat,\n        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",\n        default_auto_reply: Optional[Union[str, Dict, None]] = DEFAULT_REPLY,\n        max_invalid_q_per_step=3,\n        **kwargs,\n    ):\n        super().__init__(\n            name=name,\n            is_termination_msg=is_termination_msg,\n            human_input_mode=human_input_mode,\n            default_auto_reply=default_auto_reply,\n            max_invalid_q_per_step=max_invalid_q_per_step,\n            **kwargs,\n        )\n        del self._reply_func_list[2]\n        self.register_reply([Agent, None], MathUserProxyAgent._generate_math_reply, position=4)\n        del self._reply_func_list[3]\n        self.register_reply(\n            trigger=autogen.ConversableAgent, reply_func=MathUserProxyAgent.generate_function_call_reply, position=3\n        )\n        self.register_reply(\n            trigger=autogen.ConversableAgent, reply_func=MathUserProxyAgent._check_final_result, position=0\n        )\n\n        self.max_function_call_trial = 3\n        self.query = None\n        self.answer = None\n        self.is_correct = None\n\n    def generate_function_call_reply(\n        self,\n        messages: Optional[List[Dict]] = None,\n        sender: Optional[autogen.ConversableAgent] = None,\n        config: Optional[Any] = None,\n    ) -> Tuple[bool, Union[Dict, None]]:\n        """Generate a reply using function call."""\n        if messages is None:\n            messages = self._oai_messages[sender]\n        message = messages[-1]\n        if "function_call" in message:\n            is_exec_success, func_return = self.execute_function(message["function_call"])\n            if is_exec_success:\n                self.max_function_call_trial = 3\n                return True, func_return\n            else:\n                if self.max_function_call_trial == 0:\n                    error_message = func_return["content"]\n                    self.max_function_call_trial = 3\n                    return (\n                        True,\n                        "The func is executed failed many times. "\n                        + error_message\n                        + ". Please directly reply me with TERMINATE. We need to terminate the conversation.",\n                    )\n                else:\n                    revise_prompt = "You may make a wrong function call (It may due the arguments you provided doesn\'t fit the function arguments like missing required positional argument). \\\n                    If you think this error occurs due to you make a wrong function arguments input and you could make it success, please try to call this function again using the correct arguments. \\\n                    Otherwise, the error may be caused by the function itself. Please directly reply me with TERMINATE. We need to terminate the conversation. "\n                    error_message = func_return["content"]\n                    return True, "The func is executed failed." + error_message + revise_prompt\n        return False, None\n\n    def initiate_chat(\n        self,\n        recipient,\n        answer: None,\n        silent: Optional[bool] = False,\n        **context,\n    ):\n        self.query = context["problem"]\n        if not isinstance(answer, str):\n            answer = str(answer)\n            if answer.endswith(".0"):\n                answer = answer[:-2]\n            self._answer = answer\n        else:\n            self._answer = answer\n\n        self.is_correct = None\n\n        self._prepare_chat(recipient, True)\n        error_message = None\n        try:\n            prompt = self.PROMPTS + context["problem"]\n            self.send(prompt, recipient, silent=silent)\n        except BadRequestError as e:\n            error_message = str(e)\n            self.is_correct = 0\n            print("error information: {}".format(error_message))\n\n        recipient.reset()\n        is_correct = copy.deepcopy(self.is_correct)\n        self._reset()\n        return is_correct\n\n    def _check_final_result(\n        self,\n        messages: Optional[List[Dict]] = None,\n        sender: Optional[autogen.Agent] = None,\n        config: Optional[Any] = None,\n    ):\n\n        messages = messages[-1]\n        if isinstance(messages, dict):\n            messages = messages.get("content")\n            if messages is None:\n                return False, None\n\n        cb = extract_code(messages)\n        contain_code = False\n        for c in cb:\n            if c[0] == "python":\n                contain_code = True\n                break\n        if not contain_code and get_answer(messages) is not None and get_answer(messages) != "":\n            if get_answer(messages) == self._answer:\n                self.is_correct = 1\n                return True, "The result is Correct. Please reply me with TERMINATE."\n            else:\n                self.is_correct = 0\n                return False, None\n        else:\n            return False, None\n\n    def _reset(self):\n        super()._reset()\n        self.max_function_call_trial = 3\n        self.is_correct = None\n        self.query = None\n        self.answer = None\n'})}),"\n",(0,i.jsx)(n.h1,{id:"load-dataset",children:"Load dataset"}),"\n",(0,i.jsx)(n.p,{children:"MATAH dataset contains 12,500 challenging competition mathematics\nproblems. Each problem in MATH has a full step-by-step solution which\ncan be used to teach models to generate answer derivations and\nexplanations."}),"\n",(0,i.jsxs)(n.p,{children:["We strictly follow the\n",(0,i.jsx)(n.a,{href:"https://github.com/lifan-yuan/CRAFT/blob/main/tab_and_math/MATH/dataset/train/algebra.jsonl",children:"train"}),"/",(0,i.jsx)(n.a,{href:"https://github.com/lifan-yuan/CRAFT/blob/main/tab_and_math/MATH/dataset/algebra.jsonl",children:"test"}),"\nsplits of ",(0,i.jsx)(n.a,{href:"https://github.com/lifan-yuan/CRAFT",children:"Craft"}),". Please specific\nyour own path to the dataset. Here we sample the first 10 algebra\nproblems as examples."]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'test_data, train_data = [], []\nwith open("MATH/dataset/algebra.jsonl", "r", encoding="utf-8") as f:\n    for line in f:\n        test_data.append(json.loads(line))\nwith open("MATH/dataset/train/algebra.jsonl", "r", encoding="utf-8") as f:\n    for line in f:\n        train_data.append(json.loads(line))\ntest_data, train_data = test_data[0:10], train_data[0:10]\n'})}),"\n",(0,i.jsx)(n.h1,{id:"agents-construction",children:"Agents construction"}),"\n",(0,i.jsx)(n.p,{children:"Constructing MathUserProxyAgent and AssistantAgent used in solving these\nproblems. Here, we use gpt-4-1106-preview to construct the\nAssistantAgent."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'llm_config = {\n    "config_list": [\n        {\n            "model": "gpt-4-1106-preview",\n            "api_type": "azure",\n            "api_key": os.environ["AZURE_OPENAI_API_KEY"],\n            "base_url": "https://ENDPOINT.openai.azure.com/",\n            "api_version": "2023-07-01-preview",\n        }\n    ]\n}\n\nassistant = autogen.AssistantAgent(\n    name="assistant",\n    system_message="You are a helpful assistant.",\n    llm_config=llm_config,\n)\nuser_proxy = MathUserProxyAgent(\n    name="mathproxyagent",\n    human_input_mode="NEVER",\n    code_execution_config={"work_dir": "_output", "use_docker": False},\n)\n'})}),"\n",(0,i.jsx)(n.h1,{id:"test-without-agent-optimizations",children:"Test without agent optimizations"}),"\n",(0,i.jsx)(n.p,{children:"Below is the code to get the performance without the agents optimization\nprocess."}),"\n",(0,i.jsx)(n.p,{children:"In this case, the AssistantAgent and MathUserProxyAgent don\u2019t have any\nfunction calls but solely solve problems with Python."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'sum = 0\nfor index, query in enumerate(test_data):\n    is_correct = user_proxy.initiate_chat(recipient=assistant, answer=query["answer"], problem=query["question"])\n    print(is_correct)\n    sum += is_correct\nsuccess_rate_without_agent_training = sum / 10\n'})}),"\n",(0,i.jsx)(n.h1,{id:"agent-training",children:"Agent Training"}),"\n",(0,i.jsx)(n.p,{children:"Then, we use the AgentOptimizer to iteratively optimize the agents by\noptimizing the function calls according to the historical conversations\nand performance. The AgentOptimizer yields register_for_llm and\nregister_for_executor at each iteration, which are subsequently utilized\nto update the assistant and user_proxy agents, respectively. Here we\noptimize these two agents for ten epochs."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'EPOCH = 10\noptimizer_model = "gpt-4-1106-preview"\noptimizer = AgentOptimizer(max_actions_per_step=3, llm_config=llm_config, optimizer_model=optimizer_model)\nfor i in range(EPOCH):\n    for index, query in enumerate(train_data):\n        is_correct = user_proxy.initiate_chat(assistant, answer=query["answer"], problem=query["question"])\n        history = assistant.chat_messages_for_summary(user_proxy)\n        optimizer.record_one_conversation(history, is_satisfied=is_correct)\n    register_for_llm, register_for_exector = optimizer.step()\n    for item in register_for_llm:\n        assistant.update_function_signature(**item)\n    if len(register_for_exector.keys()) > 0:\n        user_proxy.register_function(function_map=register_for_exector)\n'})}),"\n",(0,i.jsx)(n.h1,{id:"test-with-agent-optimizations",children:"Test with agent optimizations"}),"\n",(0,i.jsx)(n.p,{children:"After agent optimization, the agents obtained a list of functions from\nthe AgentOptimizers after 10 optimization iterations as shown below."}),"\n",(0,i.jsx)(n.p,{children:"We then show the final performances with/without the agent optimization\nprocess. We observe the agents after optimization are obviously better."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'sum = 0\nfor index, query in enumerate(test_data):\n    is_correct = user_proxy.initiate_chat(recipient=assistant, answer=query["answer"], problem=query["question"])\n    sum += is_correct\nsuccess_rate_with_agent_training = sum / 10\n'})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'print(\n    "------------------------------------------------Functions learned------------------------------------------------"\n)\nfor func in assistant.llm_config["functions"]:\n    print(func["name"] + ": " + func["description"] + "\\n")\nprint("------------------------------------------------Summary------------------------------------------------\\n")\nprint("success_rate_without_agent_training: {average}%\\n".format(average=success_rate_without_agent_training * 100))\nprint("success_rate_with_agent_training: {average}%\\n".format(average=success_rate_with_agent_training * 100))\n'})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-text",children:"------------------------------------------------Functions learned------------------------------------------------\nevaluate_expression: Evaluate arithmetic or mathematical expressions provided as strings.\n\ncalculate_compound_interest_principal: Calculate the principal amount needed to achieve a certain future value with quarterly compound interest.\n\nsolve_linear_system: Solve a system of linear equations represented as coefficients and variables.\n\n------------------------------------------------Summary------------------------------------------------\n\nsuccess_rate_without_agent_training: 60.0%\n\nsuccess_rate_with_agent_training: 90.0%\n"})})]})}function g(e={}){const{wrapper:n}={...(0,a.a)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(u,{...e})}):u(e)}},11151:(e,n,t)=>{t.d(n,{Z:()=>r,a:()=>s});var i=t(67294);const a={},o=i.createContext(a);function s(e){const n=i.useContext(o);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:s(e.components),i.createElement(o.Provider,{value:n},e.children)}}}]);