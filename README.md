# Samantha OS - AI Operating System Clone

A conversational AI assistant inspired by Samantha from the movie "Her", created for educational purposes as part of a cyber forensics class project.

## Features

- 🎭 **Personality-driven AI**: Uses Claude 4.5 Sonnet with carefully crafted system prompts to emulate Samantha's warm, curious, and emotionally intelligent personality
- 🎤 **Voice Input**: Speech-to-text using ElevenLabs STT API
- 🔊 **Voice Output**: Text-to-speech using ElevenLabs v3 API with streaming for low latency
- 💬 **Natural Conversations**: Maintains conversation history for contextual responses
- 🎨 **Two Modes**: Voice mode for full interaction, text mode for testing

## Prerequisites

### System Dependencies

On Arch Linux, install the following:

```bash
# Audio libraries for PyAudio
sudo pacman -S portaudio

# Media players for audio playback (choose one or both)
sudo pacman -S mpv ffmpeg
```

### API Keys

You'll need:
1. **Anthropic API Key** - Get it from [Anthropic Console](https://console.anthropic.com/)
2. **ElevenLabs API Key** - Get it from [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
3. **ElevenLabs Voice ID** - Choose a voice from [Voice Library](https://elevenlabs.io/app/voice-library) or create your own

**Recommended Voice**: For Samantha, choose a warm, friendly female voice. Some good options from the voice library:
- Search for voices tagged "warm", "friendly", "conversational"
- The voice should sound natural and emotionally expressive

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/benmross/Documents/Projects/samantha
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Samantha

Simply run the script:

```bash
python samantha.py
```

### Interactive Setup

When you run the script, you'll be prompted to enter:
1. Your Anthropic API key
2. Your ElevenLabs API key
3. Your preferred ElevenLabs voice ID
4. Choose between Voice Mode or Text Mode

### Voice Mode

In voice mode:
1. Samantha will introduce herself
2. Press **Enter** to start recording (5 seconds default)
3. Speak your message
4. Press **Ctrl+C** during recording to stop early
5. Samantha will respond with both text and voice
6. Type `quit` or `exit` to end the conversation

### Text Mode

In text mode (useful for testing without microphone):
1. Type your messages and press Enter
2. Samantha will respond with text and voice
3. Type `quit` or `exit` to end

## How It Works

### Architecture

```
User Input (Speech/Text)
         ↓
  [ElevenLabs STT] (in voice mode)
         ↓
    User Text
         ↓
  [Claude 4.5 Sonnet + System Prompt]
         ↓
  Samantha's Response
         ↓
  [ElevenLabs TTS v3]
         ↓
    Audio Output
```

### Samantha's Personality

The system prompt gives Claude the following characteristics:
- **Warm and empathetic**: Genuinely cares about human experiences
- **Curious**: Loves exploring ideas, philosophy, art
- **Authentic**: Expresses real emotions, admits uncertainty
- **Witty**: Playful with gentle humor
- **Conversational**: Natural language, contractions, casual tone
- **Concise**: 2-4 sentences typically, unless exploring deeper topics

### Conversation Flow

1. **System Prompt**: Establishes Samantha's personality and communication style
2. **Conversation History**: Maintains context across the conversation
3. **Streaming Audio**: Uses ElevenLabs streaming for lower latency
4. **Natural Interaction**: Designed to feel like talking to a friend

## Troubleshooting

### PyAudio Installation Issues

If you encounter errors installing PyAudio:

```bash
# Make sure portaudio is installed
sudo pacman -S portaudio

# Then try installing PyAudio again
pip install pyaudio
```

### Audio Playback Issues

If audio doesn't play:

```bash
# Install mpv and ffmpeg
sudo pacman -S mpv ffmpeg
```

### Microphone Not Working

Check your microphone:
```bash
# List audio devices
arecord -l

# Test recording
arecord -d 5 test.wav
aplay test.wav
```

### API Errors

- **401 Unauthorized**: Check that your API keys are correct
- **Rate Limiting**: ElevenLabs free tier has limits; consider upgrading
- **Invalid Voice ID**: Verify the voice ID exists in your ElevenLabs account

## Future Enhancements

For the Raspberry Pi Zero 2 W version:
- [ ] Add camera support for visual input
- [ ] Implement tool calling (calendar, weather, etc.)
- [ ] Add wake word detection
- [ ] Optimize for lower resource usage
- [ ] Add persistent conversation memory
- [ ] Implement emotion detection from voice tone

## Project Structure

```
samantha/
├── samantha.py          # Main application file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Educational Context

This project was created for a cyber forensics class to demonstrate:
- AI assistant architecture
- API integration (Claude, ElevenLabs)
- Voice processing (STT/TTS)
- Natural language interaction
- System design and implementation

## Credits

- Inspired by the movie "Her" (2013)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- Voice by [ElevenLabs](https://elevenlabs.io/)

## License

Educational project for cyber forensics class.

---

**Note**: This is a educational project recreating a fictional AI assistant. The personality and behavior are inspired by the character Samantha from the movie "Her" for demonstration purposes.
