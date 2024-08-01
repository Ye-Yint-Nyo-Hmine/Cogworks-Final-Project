from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from utils.cryptographer import encrypt, decrypt
#from utils.sys_prompt import SYS_PROMPT as sys_prompt
#from utils.sys_prompt import KEY
from profiles.Profile import Profile
from multiprocessing import Process

def export_sys_prompt(prompt, PATH=r"LLM\utils\sys.txt"):
    with open(PATH, "w") as f:
        f.write(prompt)
        f.close()
    
def load_sys_prompt(PATH=r"LLM\utils\sys.txt"):
    with open(PATH, "r") as f:
        read = f.read()
        f.close()
        return read

def beta_load_sys_prompt(sys_prompt, KEY):
    encoded = encrypt(sys_prompt, KEY)
    export_sys_prompt(encoded)
    print("Loading ...")
    loaded = load_sys_prompt()
    decoded = decrypt(loaded, KEY)
    return decoded


def initialize_profile(name, user_instr=None, response_instr=None):
    profile = Profile(name)
    if user_instr is not None and response_instr is not None:
        profile.custom_instructions(user_instr, response_instr)
    profile.update_profile()    
    return profile


# non host only
load_sys_prompts = load_sys_prompt()
KEY = "TLJRVBFrZY-0YVoflosslIcv8volnQTdSl940c"
sys_prompt = decrypt(load_sys_prompts, KEY)


template = """ 
Current date: {time}
{sys_prompt}

User profile:
{user_profile}

Here is the conversation history: 
{context}


User: {question}
"""



model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def run_chatbot(profile, context):
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            context += "\nUser: Exited ...\n\nUser has exited the chat. New session will began."
            profile.update_history(context)
            break

        result = chain.invoke({"time": datetime.today(), 
                               "sys_prompt": sys_prompt,
                               "user_profile": profile.try_load_profile(),
                               "context": context, 
                               "question": user_input})
        print(f"ByteGPT: {result}")
        try:
            if "EVENT-ALERT" in user_input:
                context += f"\nSystem: {user_input}\nMe: {result}"
            else:
                context += f"\nUser: {user_input}\nMe: {result}"
        except:
            continue


if __name__ == "__main__":
    print("Started")
    user_name = input("Enter name: ") 
    custom_instr = True if (input("Do you want to include/update specific information about yourself and home? (Y/n): ")).lower() == "y" else False
    if custom_instr:
        user_instr = input("Enter user info/instructions: ")
        response_instr = input("Enter home info/instructions: ")
        profile = initialize_profile(user_name, user_instr, response_instr)
    else:
        profile = initialize_profile(user_name)
    
    context = profile.try_load_history()
    
    run_chatbot(profile, context)
    