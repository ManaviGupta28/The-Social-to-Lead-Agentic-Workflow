# 🚀 AutoStream AI Agent - Quick Start

## ✅ Installation Complete!

All dependencies have been installed successfully.

## 🎯 Next Steps

### 1. Start the Backend Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### 2. Open the Frontend

Open `index.html` in your browser:
- **Double-click** the file, OR
- **Right-click** → Open with → Your browser, OR
- **Drag and drop** into browser window

### 3. Start Chatting!

Try these example conversations:

**Example 1: Learn About Pricing**
```
You: Hi, what are your pricing plans?
Agent: [Explains Basic and Pro plans with features]
```

**Example 2: Sign Up Flow**
```
You: I want to try the Pro plan for my YouTube channel
Agent: That's great! Can I get your name first?
You: John Doe
Agent: Thanks, John! What's your email address?
You: john@example.com  
Agent: Great! Which platform do you primarily create content for?
You: YouTube
Agent: Perfect! I've got you all set up... [Confirmation]
```

**Check Server Console** for lead capture confirmation:
```
🎯 Lead captured successfully: John Doe, john@example.com, YouTube
```

## 🧪 Automated Testing

Run the test script:
```bash
python test_conversation.py
```

## 📚 Documentation

- **README.md** - Full documentation and setup guide
- **QUICK START.md** - This file
- **walkthrough.md** - Complete project walkthrough (in .gemini folder)

## 🆘 Troubleshooting

**Server won't start?**
- Check that port 8000 is not in use
- Verify `.env` file has your API key

**Frontend can't connect?**
- Ensure backend is running first
- Check console for CORS errors (should be fixed)

**Embedding errors?**
- First run will download the model (one-time, ~100MB)
- Takes 30-60 seconds on first startup

## 🎉 You're All Set!

The AutoStream AI Agent is ready to:
- Answer questions about pricing and features
- Detect high-intent leads
- Capture lead information
- Execute backend actions

Enjoy! 🚀
