
class Profile:
    def __init__(self, name):
        self.name = name
        self.user_instr = f"My name is {self.name}"
        self.response_instr = ""
        self.temp_bio = []
    
    def update_profile(self):
        with open(f"LLM\profiles\{self.name}.txt", "w") as f:
            if self.response_instr != "":
                f.write(f"\n{self.user_instr}\nThe user provided the additional info about how they would like you to respond:\n{self.response_instr}")
            else:
                f.write(f"\n{self.user_instr}")
            f.close()
    
    def load_profile(self):
        with open(f"LLM\profiles\{self.name}.txt", "r") as f:
            read = f.read()
            f.close()
            return read
    
    def create_history(self):
        with open(f"LLM\history\{self.name}_history.txt", "w") as f:
            f.close()
    
    def update_history(self, context):
        with open(f"LLM\history\{self.name}_history.txt", "a") as f:
            f.write(f"{context}")
            f.close()
        
    def load_history(self):
        with open(f"LLM\history\{self.name}_history.txt", "r") as f:
            read = f.read()
            f.close()
            return read
    
    def tobio(self, msg):
        self.temp_bio.append(msg)
        
    def load_tobio(self):
        return self.temp_bio
    
    def try_load_profile(self):
        try:
            return self.load_profile()
        except FileNotFoundError:
            return False
        
    def try_load_history(self):
        try:
            return self.load_history()
        except FileNotFoundError:
            self.create_history()
            return self.load_history()

    
    def custom_instructions(self, user_instr, response_instr):
        self.user_instr = user_instr
        self.response_instr = response_instr
    
