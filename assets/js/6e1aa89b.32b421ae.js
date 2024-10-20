"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[2713],{40446:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>a,contentTitle:()=>l,default:()=>p,frontMatter:()=>r,metadata:()=>t,toc:()=>c});var s=o(85893),i=o(11151);const r={},l="Non-OpenAI Models",t={id:"topics/non-openai-models/about-using-nonopenai-models",title:"Non-OpenAI Models",description:"AutoGen allows you to use non-OpenAI models through proxy servers that provide",source:"@site/docs/topics/non-openai-models/about-using-nonopenai-models.md",sourceDirName:"topics/non-openai-models",slug:"/topics/non-openai-models/about-using-nonopenai-models",permalink:"/autogen/docs/topics/non-openai-models/about-using-nonopenai-models",draft:!1,unlisted:!1,editUrl:"https://github.com/autogenhub/autogen/edit/main/website/docs/topics/non-openai-models/about-using-nonopenai-models.md",tags:[],version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"Using Transform Messages during Speaker Selection",permalink:"/autogen/docs/topics/groupchat/transform_messages_speaker_selection"},next:{title:"Tips for Non-OpenAI Models",permalink:"/autogen/docs/topics/non-openai-models/best-tips-for-nonopenai-models"}},a={},c=[{value:"OpenAI-compatible API proxy server",id:"openai-compatible-api-proxy-server",level:2},{value:"Cloud-based proxy servers",id:"cloud-based-proxy-servers",level:3},{value:"Locally run proxy servers",id:"locally-run-proxy-servers",level:3},{value:"Configuration for Non-OpenAI models",id:"configuration-for-non-openai-models",level:3},{value:"Custom Model Client class",id:"custom-model-client-class",level:2}];function d(e){const n={a:"a",admonition:"admonition",h1:"h1",h2:"h2",h3:"h3",img:"img",li:"li",p:"p",ul:"ul",...(0,i.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h1,{id:"non-openai-models",children:"Non-OpenAI Models"}),"\n",(0,s.jsxs)(n.p,{children:["AutoGen allows you to use non-OpenAI models through proxy servers that provide\nan OpenAI-compatible API or a ",(0,s.jsx)(n.a,{href:"https://autogenhub.github.io/autogen/blog/2024/01/26/Custom-Models",children:"custom model client"}),"\nclass."]}),"\n",(0,s.jsx)(n.p,{children:"Benefits of this flexibility include access to hundreds of models, assigning specialized\nmodels to agents (e.g., fine-tuned coding models), the ability to run AutoGen entirely\nwithin your environment, utilising both OpenAI and non-OpenAI models in one system, and cost\nreductions in inference."}),"\n",(0,s.jsx)(n.h2,{id:"openai-compatible-api-proxy-server",children:"OpenAI-compatible API proxy server"}),"\n",(0,s.jsxs)(n.p,{children:["Any proxy server that provides an API that is compatible with ",(0,s.jsx)(n.a,{href:"https://platform.openai.com/docs/api-reference",children:"OpenAI's API"}),"\nwill work with AutoGen."]}),"\n",(0,s.jsx)(n.p,{children:"These proxy servers can be cloud-based or running locally within your environment."}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{alt:"Cloud or Local Proxy Servers",src:o(75797).Z+"",width:"400",height:"308"})}),"\n",(0,s.jsx)(n.h3,{id:"cloud-based-proxy-servers",children:"Cloud-based proxy servers"}),"\n",(0,s.jsx)(n.p,{children:"By using cloud-based proxy servers, you are able to use models without requiring the hardware\nand software to run them."}),"\n",(0,s.jsxs)(n.p,{children:["These providers can host open source/weight models, like ",(0,s.jsx)(n.a,{href:"https://huggingface.co/",children:"Hugging Face"}),"\nand ",(0,s.jsx)(n.a,{href:"https://mistral.ai/",children:"Mistral AI"}),",\nor their own closed models."]}),"\n",(0,s.jsxs)(n.p,{children:["When cloud-based proxy servers provide an OpenAI-compatible API, using them in AutoGen\nis straightforward. With ",(0,s.jsx)(n.a,{href:"/docs/topics/llm_configuration",children:"LLM Configuration"})," done in\nthe same way as when using OpenAI's models, the primary difference is typically the\nauthentication which is usually handled through an API key."]}),"\n",(0,s.jsx)(n.p,{children:"Examples of using cloud-based proxy servers providers that have an OpenAI-compatible API\nare provided below:"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:(0,s.jsx)(n.a,{href:"/docs/topics/non-openai-models/cloud-togetherai",children:"Together AI example"})}),"\n",(0,s.jsx)(n.li,{children:(0,s.jsx)(n.a,{href:"/docs/topics/non-openai-models/cloud-mistralai",children:"Mistral AI example"})}),"\n",(0,s.jsx)(n.li,{children:(0,s.jsx)(n.a,{href:"/docs/topics/non-openai-models/cloud-anthropic",children:"Anthropic Claude example"})}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"locally-run-proxy-servers",children:"Locally run proxy servers"}),"\n",(0,s.jsx)(n.p,{children:"An increasing number of LLM proxy servers are available for use locally. These can be\nopen-source (e.g., LiteLLM, Ollama, vLLM) or closed-source (e.g., LM Studio), and are\ntypically used for running the full-stack within your environment."}),"\n",(0,s.jsx)(n.p,{children:"Similar to cloud-based proxy servers, as long as these proxy servers provide an\nOpenAI-compatible API, running them in AutoGen is straightforward."}),"\n",(0,s.jsx)(n.p,{children:"Examples of using locally run proxy servers that have an OpenAI-compatible API are\nprovided below:"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:(0,s.jsx)(n.a,{href:"/docs/topics/non-openai-models/local-litellm-ollama",children:"LiteLLM with Ollama example"})}),"\n",(0,s.jsx)(n.li,{children:(0,s.jsx)(n.a,{href:"/docs/topics/non-openai-models/local-lm-studio",children:"LM Studio"})}),"\n",(0,s.jsx)(n.li,{children:(0,s.jsx)(n.a,{href:"/docs/topics/non-openai-models/local-vllm",children:"vLLM example"})}),"\n"]}),"\n",(0,s.jsx)(n.admonition,{type:"tip",children:(0,s.jsx)(n.p,{children:"If you are planning to use Function Calling, not all cloud-based and local proxy servers support\nFunction Calling with their OpenAI-compatible API, so check their documentation."})}),"\n",(0,s.jsx)(n.h3,{id:"configuration-for-non-openai-models",children:"Configuration for Non-OpenAI models"}),"\n",(0,s.jsxs)(n.p,{children:["Whether you choose a cloud-based or locally-run proxy server, the configuration is done in\nthe same way as using OpenAI's models, see ",(0,s.jsx)(n.a,{href:"/docs/topics/llm_configuration",children:"LLM Configuration"}),"\nfor further information."]}),"\n",(0,s.jsxs)(n.p,{children:["You can use ",(0,s.jsx)(n.a,{href:"/docs/topics/llm_configuration#config-list-filtering",children:"model configuration filtering"}),"\nto assign specific models to agents."]}),"\n",(0,s.jsx)(n.h2,{id:"custom-model-client-class",children:"Custom Model Client class"}),"\n",(0,s.jsx)(n.p,{children:"For more advanced users, you can create your own custom model client class, enabling\nyou to define and load your own models."}),"\n",(0,s.jsxs)(n.p,{children:["See the ",(0,s.jsx)(n.a,{href:"/blog/2024/01/26/Custom-Models",children:"AutoGen with Custom Models: Empowering Users to Use Their Own Inference Mechanism"}),"\nblog post and ",(0,s.jsx)(n.a,{href:"/docs/notebooks/agentchat_custom_model/",children:"this notebook"})," for a guide to creating custom model client classes."]})]})}function p(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},75797:(e,n,o)=>{o.d(n,{Z:()=>s});const s=o.p+"assets/images/cloudlocalproxy-f2c174642c247a91eeca30ac29b01fcf.png"},11151:(e,n,o)=>{o.d(n,{Z:()=>t,a:()=>l});var s=o(67294);const i={},r=s.createContext(i);function l(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function t(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:l(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);