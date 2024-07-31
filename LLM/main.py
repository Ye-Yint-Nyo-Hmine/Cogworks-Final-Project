from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from utils.cryptographer import encrypt, decrypt
#from utils.sys_prompt import SYS_PROMPT as sys_prompt
#from utils.sys_prompt import KEY
from profiles.Profile import Profile

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


def report_event(event):
    """
    This is the function which you can deploy to integrate and implement other functions
    
    Args:
    event: str
            Event can be anything;  (ie: 'Unrecognized face detected', 'Known face detected, Matched: User name, Accuracy: X%',
                                    'Stranger detected near home', 'Baby crying detected', 'Fire detected')
            Events can also specify where it happened: such as 'near rear part of the house', 'near front door', 'near the garage'
            
    Returns:
        doesn't return anything when called,
        but does report the user of the event and function accordingly.
        
    eg:
    
    # Note: Remember that this function can be called in your own files, it does not need to be called inside this main function
    
    if baby_monitor.baby_detected() == True:
        report_event("Baby crying detected near front door")
    if fire_detection == True:
        report_event("Fire has been detected")
        
    You don't need to add anything to the function, but just need to call it
    
    """
    
    pass
    
    


def tokenize_report(prompt):
    if 'eventreport' in prompt:
        try:
            
        except:
            print("Error occurred while tokenizing report")

def checkReport(msg):
    tokenized_report = tokenize_report()


def main():
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
    main()
