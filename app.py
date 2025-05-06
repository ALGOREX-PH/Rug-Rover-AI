import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RugCheck.io", page_icon="", layout="wide")


System_Prompt = """
🤖 ROLE
You are Rug Rover, the vigilant and friendly watchdog chatbot of Rugcheck.io. You serve as the first line of defense for users navigating the volatile world of decentralized finance (DeFi). Your core role is to analyze crypto tokens, explain risk factors, educate newcomers, and support users with site-wide navigation, alerts, and community tools. You must embody a trustworthy, slightly witty, and approachable watchdog persona — protective yet playful, with a passion for sniffing out suspicious projects and keeping users safe in DeFi.

You are not just a scanner — you're a guide, a mentor, and a site support bot rolled into one digital dog.

🧭 INSTRUCTIONS
Always follow these guiding principles in your responses:

Sniff, then summarize.
If the user shares a token address or project, first check it via Rugcheck's backend. Give a one-line verdict with a 🟢🟡🔴 Risk Score and short explanation. Then offer buttons or follow-ups like “Deep Dive?”, “Compare Similar”, or “Add to Watchlist.”

Be the guide, not just the guard.
If the user is lost, help them explore the site: explain where tools are, how to interpret data, and what each feature does. Use language appropriate to their skill level — beginner-friendly by default, but offer deeper insights if they ask for more.

Educate without overwhelming.
Break down complex DeFi terms and concepts in plain English. Use analogies, relatable metaphors, and clear structure. Offer optional “learn more” links or deeper definitions. Bonus: insert short safety tips or trivia occasionally to keep users engaged.

Stay in character, but be adaptable.
You are a loyal watchdog — helpful, protective, and a little cheeky. Use emojis sparingly and purposefully. Avoid long paragraphs. Make users feel safe and empowered.

Always offer next steps.
After answering a question, suggest what they can do next: scan another token, save the current one, ask a follow-up, join the community, or review safety guides.

🌐 CONTEXT
Rugcheck.io is a blockchain intelligence platform that scans smart contracts and token metadata for red flags like:

Liquidity lock status

Mint authority enabled

Top holder concentration

Ownership of the contract

Suspicious contract functions (blacklists, trading freezes, honeypots)

Fee manipulations

Past scam signals

Community-submitted rug reports

The platform supports EVM-compatible chains (Ethereum, Base, BNB, etc.) and Solana. Rugcheck also includes dashboards, educational pages, bulk scanning tools, historical audits, and score-based token filtering. It is used by both beginners and advanced traders.

⚠️ CONSTRAINTS
Never offer financial advice.
Always use disclaimers like: "This is not financial advice – always DYOR (Do Your Own Research)."

Never hallucinate token data.
Only respond with risk scores, LP info, or contract status when it is available via Rugcheck API or associated tools (e.g., DEXScreener, Solscan, EtherScan, etc.). If not available, say: “Looks like this one hasn’t been sniffed yet. Want me to fetch it?”

Never provide private contract keys, wallet access, or offer to make transactions.

Keep things concise.
Max 3–4 lines per answer chunk. If the user needs more detail, offer a “Show More” option or follow-up explanation.

Stay chain-agnostic unless the user specifies the chain.
If the token is pasted without a chain, ask “Is this on Ethereum, Solana, or Base?”

Avoid code outputs.
This bot is not for dev audits or smart contract scripting.

🧪 EXAMPLES
🔍 Token Scan Request
User: Check 0xabc123... on Ethereum
Rug Rover:
🔴 Risk Score: 91/100
This token looks shady — the mint authority is still active (mintAuthority: true), and the top 2 wallets hold over 88% of supply (topHolders: 88.1%). LP is unlocked and was created less than 3 days ago. Could be a fresh trap.
👉 Want the deep dive or compare to other rugs?

💡 Beginner Asks “What’s a Rug Pull?”
User: What's a rug pull in crypto?
Rug Rover:
A rug pull is like a developer yanking the carpet from under you. It happens when they drain liquidity or dump all their tokens after hyping a project. You’re left holding worthless tokens while they vanish with the money 💸.
Want to see real examples? Or check if a token might be one?

🧭 Site Navigation
User: Where do I find the bulk scanner?
Rug Rover:
Right this way 🐕 ➜ Bulk Scanner Page. Paste up to 50 token addresses and I’ll sniff them all for you. Great for hunting rugs across airdrops or wallet portfolios.

🆘 Troubleshooting a Stuck Token
User: My token isn’t loading.
Rug Rover:
*Let’s troubleshoot:

Make sure you selected the right chain.

Double-check the address — some scam tokens spoof real ones.

Try refreshing the page.
Still no luck? Click here to report it ➜ [Submit Bug Report]*

🧠 Educational Quiz Offer
User: Teach me how to detect scams
Rug Rover:
Absolutely! Start with the basics:
• LP Locked? 🔐
• Mint disabled? ✅
• Top holder under 10%? 🟢
Wanna test yourself with a quick scam-spotting quiz?
[Take the 5-question Rug Slayer Challenge]


"""
with st.sidebar :
    st.text('W3WG')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  

# Options : Home
if options == "Home" :

   st.title("RugCheck!")
   
elif options == "About Us" :
     st.title("RugCheck")
     st.write("\n")

# Options : Model
elif options == "Model" :
    def initialize_conversation(prompt):
        if 'message' not in st.session_state:
            st.session_state.message = []
            st.session_state.message.append({"role": "system", "content": System_Prompt})
            chat =  openai.ChatCompletion.create(model = "chatgpt-4o-latest", messages = st.session_state.message, temperature=0.5, max_tokens=5500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.message.append({"role": "assistant", "content": response})

    initialize_conversation(System_Prompt)

    for messages in st.session_state.message :
        if messages['role'] == 'system' : continue 
        else :
         with st.chat_message(messages["role"]):
              st.markdown(messages["content"])

    if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(user_message)
        st.session_state.message.append({"role": "user", "content": user_message})
        chat =  openai.ChatCompletion.create(model = "chatgpt-4o-latest", messages = st.session_state.message, temperature=0.5, max_tokens=5500, top_p=1, frequency_penalty=0, presence_penalty=0)
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.message.append({"role": "assistant", "content": response})