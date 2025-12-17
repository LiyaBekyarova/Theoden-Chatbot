from tkinter import *
from tkinter import font as tkfont
from chat import get_response, bot_name

BG_WINDOW = "#1a3d1a"
BG_HEADER = "#8b4513"
BG_CHAT = "#fffacd"
TEXT_CHAT = "#2f1810"
TEXT_HEADER = "#ffd700"
USER_COLOR = "#228b22"
BOT_COLOR = "#b8860b"
ENTRY_BG = "#fff8dc"
BUTTON_BG = "#cd853f"
BUTTON_ACTIVE = "#a0522d"
BORDER_GOLD = "#daa520"

FONT_HEADER = ("Georgia", 18, "bold")
FONT_MESSAGE = ("Georgia", 11)
FONT_SENDER = ("Georgia", 11, "bold")
FONT_ENTRY = ("Georgia", 12)
FONT_BUTTON = ("Georgia", 12, "bold")

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("⚔️ Théoden - King of Rohan ⚔️")
        self.window.geometry("650x750")
        self.window.configure(bg=BG_WINDOW)
        
        top_border = Frame(self.window, bg=BORDER_GOLD, height=4)
        top_border.pack(fill=X)
        
        header_frame = Frame(self.window, bg=BG_HEADER, height=90)
        header_frame.pack(fill=X)
        header_frame.pack_propagate(False)
        
        title_label = Label(header_frame, text="👑 THÉODEN 👑", 
                           font=FONT_HEADER, bg=BG_HEADER, fg=TEXT_HEADER)
        title_label.pack(pady=(10, 2))
        
        subtitle = Label(header_frame, text="⚔️ King of the Mark • Lord of the Rohirrim ⚔️",
                        font=("Georgia", 10, "italic"), bg=BG_HEADER, fg=BORDER_GOLD)
        subtitle.pack()
        
        bottom_border = Frame(self.window, bg=BORDER_GOLD, height=4)
        bottom_border.pack(fill=X)
        
        chat_outer = Frame(self.window, bg=BG_WINDOW)
        chat_outer.pack(fill=BOTH, expand=True, padx=8, pady=8)
        
        chat_container = Frame(chat_outer, bg=BG_CHAT, relief=RIDGE, borderwidth=3)
        chat_container.pack(fill=BOTH, expand=True)
        
        scrollbar = Scrollbar(chat_container, bg=BG_CHAT, troughcolor=ENTRY_BG)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.text_widget = Text(chat_container, 
                                bg=BG_CHAT, 
                                fg=TEXT_CHAT,
                                font=FONT_MESSAGE, 
                                wrap=WORD,
                                yscrollcommand=scrollbar.set,
                                padx=15, 
                                pady=15,
                                relief=FLAT,
                                state=DISABLED)
        self.text_widget.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.text_widget.yview)
        
        self.text_widget.tag_configure("user", foreground=USER_COLOR, font=FONT_SENDER)
        self.text_widget.tag_configure("bot", foreground=BOT_COLOR, font=FONT_SENDER)
        self.text_widget.tag_configure("message", foreground=TEXT_CHAT, font=FONT_MESSAGE, 
                                       spacing1=3, spacing3=12)
        
        separator = Frame(self.window, bg=BORDER_GOLD, height=3)
        separator.pack(fill=X)
        
        input_outer = Frame(self.window, bg=BG_WINDOW)
        input_outer.pack(fill=X, padx=8, pady=8)
        
        input_frame = Frame(input_outer, bg=BG_HEADER, relief=RAISED, borderwidth=2)
        input_frame.pack(fill=X, padx=4, pady=4)
        
        self.msg_entry = Entry(input_frame, 
                              bg=ENTRY_BG, 
                              fg=TEXT_CHAT,
                              font=FONT_ENTRY,
                              relief=SUNKEN,
                              borderwidth=2,
                              insertbackground=TEXT_CHAT)
        self.msg_entry.pack(side=LEFT, fill=X, expand=True, ipady=10, padx=10, pady=8)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        self.send_button = Button(input_frame, 
                                  text=" SEND ", 
                                  font=FONT_BUTTON, 
                                  bg=BUTTON_BG, 
                                  fg=TEXT_CHAT,
                                  activebackground=BUTTON_ACTIVE,
                                  activeforeground=TEXT_CHAT,
                                  relief=RAISED,
                                  borderwidth=3,
                                  cursor="hand2",
                                  width=12,
                                  command=lambda: self._on_enter_pressed(None))
        self.send_button.pack(side=RIGHT, padx=10, pady=8, ipady=5)
        
        bottom_gold = Frame(self.window, bg=BORDER_GOLD, height=4)
        bottom_gold.pack(fill=X)
        
        self._add_welcome_message()
     
    def _add_welcome_message(self):
        welcome = ("⚔️ Hail, friend of Rohan! ⚔️\n\n"
                  "I am Théoden, son of Thengel, King of the Mark.\n"
                  "Speak thy mind, and let us hold counsel together.\n"
                  "The hearth is warm and the mead awaits!")
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, "👑 Théoden, King of Rohan:\n", "bot")
        self.text_widget.insert(END, welcome + "\n", "message")
        self.text_widget.configure(state=DISABLED)
    
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get().strip()
        if msg:
            self._insert_message(msg, "You")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, f"🗡️ {sender} (Rider of the Mark):\n", "user")
        self.text_widget.insert(END, msg + "\n", "message")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)
        
        response = get_response(msg)
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, f"👑 {bot_name}, King of Rohan:\n", "bot")
        self.text_widget.insert(END, response + "\n", "message")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)
        
        if msg.lower() in ["quit", "exit", "leave", "goodbye", "bye", "farewell"]:
            self.window.after(2000, self.window.destroy)
             
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()