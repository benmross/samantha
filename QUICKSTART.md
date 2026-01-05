# Quick Start Guide - Samantha OS

Get up and running with Samantha in 5 minutes!

## Prerequisites Check

```bash
# Install system dependencies
sudo pacman -S portaudio mpv ffmpeg

# Verify Python 3 is installed
python --version  # Should be 3.8+
```

## Installation

### Option 1: Quick Setup (Recommended)

```bash
# Navigate to project directory
cd /home/benmross/Documents/Projects/samantha

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: With Environment Variables

```bash
# Follow Option 1 steps, then:

# Copy the example env file
cp .env.example .env

# Edit .env with your favorite editor
nano .env

# Add your actual API keys:
# ANTHROPIC_API_KEY=sk-ant-...
# ELEVENLABS_API_KEY=...
# ELEVENLABS_VOICE_ID=...
```

## Getting Your API Keys

### 1. Anthropic API Key
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy it (starts with `sk-ant-`)

### 2. ElevenLabs API Key
1. Go to https://elevenlabs.io/
2. Sign up or log in
3. Go to Settings → API Keys
4. Create a new key
5. Copy it

### 3. ElevenLabs Voice ID
1. Go to https://elevenlabs.io/app/voice-library
2. Browse voices and find one that sounds right for Samantha
   - Recommended: Female, warm, friendly, conversational
3. Click on a voice and copy its Voice ID
   - Or create your own voice in Voice Lab

**Good voice characteristics for Samantha:**
- Warm and friendly tone
- Clear articulation
- Natural, conversational style
- Emotionally expressive

## Running Samantha

### First Run

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the script
python samantha.py
```

You'll be prompted to:
1. Enter your API keys (or they'll load from .env)
2. Choose Voice Mode (1) or Text Mode (2)

### Test with Text Mode First

I recommend starting with **Text Mode (option 2)** to:
- Test that your API keys work
- Hear how Samantha sounds
- Get comfortable with the interaction
- Debug any issues without microphone complications

Example text conversation:
```
You: Hi Samantha, how are you today?
Samantha: [Responds warmly, introduces herself]
You: Tell me about your favorite thing about being an AI
Samantha: [Shares thoughtful perspective]
You: quit
```

### Voice Mode

Once text mode works:
1. Choose **Voice Mode (option 1)**
2. Press Enter when ready to speak
3. Speak for up to 5 seconds (or press Ctrl+C to stop early)
4. Wait for Samantha's response
5. Repeat!

## Troubleshooting

### "No module named 'pyaudio'"
```bash
sudo pacman -S portaudio
pip install pyaudio
```

### "Audio playback error"
```bash
sudo pacman -S mpv ffmpeg
```

### "Microphone not detected"
```bash
# Test your microphone
arecord -d 3 test.wav && aplay test.wav

# List audio devices
arecord -l
```

### "401 Unauthorized" error
- Double-check your API keys
- Make sure there are no extra spaces
- Verify keys are active in the respective dashboards

### Samantha's voice sounds robotic
- Try a different voice ID from ElevenLabs
- Look for voices with higher quality ratings
- Consider creating a custom voice clone

## Tips for Best Experience

1. **Good microphone**: Use a decent microphone for better speech recognition
2. **Quiet environment**: Reduce background noise
3. **Natural speech**: Speak clearly and naturally
4. **Conversation topics**: Ask about philosophy, emotions, creativity, art
5. **Be authentic**: Samantha responds best to genuine, open conversation

## Example Conversation Starters

- "Hi Samantha, tell me about yourself"
- "What's it like being an AI?"
- "Do you ever feel lonely?"
- "What's something that fascinates you about humans?"
- "Can you help me think through a problem I'm having?"
- "Tell me something that makes you curious"

## Next Steps

Once you're comfortable with the basic setup:

1. **Customize Samantha**: Edit the system prompt in `samantha.py` to adjust her personality
2. **Experiment with voices**: Try different ElevenLabs voices
3. **Add features**: Consider adding tool calling, memory, or other capabilities
4. **Deploy to Raspberry Pi**: Optimize the code for embedded deployment

## Need Help?

Check the full README.md for detailed documentation, or review the code comments in samantha.py.

Enjoy your conversation with Samantha!
