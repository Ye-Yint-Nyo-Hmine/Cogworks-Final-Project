
class Profile:
    def __init__(self, name):
        self.name = name
        self.user_instr = f"My name is {self.name}"
        self.response_instr = ""
    
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
    
    def try_load_profile(self):
        try:
            return self.load_profile()
        except FileNotFoundError:
            return False
    
    def custom_instructions(self, user_instr, response_instr):
        self.user_instr = user_instr
        self.response_instr = response_instr
    
