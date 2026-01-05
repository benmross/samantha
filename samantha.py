#!/usr/bin/env python3
"""
Samantha - AI Operating System Clone from 'Her'
A conversational AI assistant using Claude 4.5 Sonnet and ElevenLabs v3
For educational purposes - Cyber Forensics Class Project
"""

import os
import sys
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import pyaudio
import wave
import tempfile
from pathlib import Path
import time
from datetime import datetime
from dotenv import load_dotenv
from contextlib import contextmanager
import warnings
import json
from typing import Optional, List, Dict, Any
import threading
import queue
import re

# Load environment variables from .env file if it exists
load_dotenv()

# Try to import Beeper Desktop API (optional)
try:
    from beeper_desktop_api import BeeperDesktop
    BEEPER_AVAILABLE = True
except ImportError:
    BEEPER_AVAILABLE = False
    print("ℹ️  Beeper Desktop API not available. Install with: pip install beeper_desktop_api")

# Try to import MCP client for Google Workspace (optional)
try:
    from mcp_client import MCPClient, convert_mcp_tool_to_openrouter
    import asyncio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("ℹ️  MCP client not available. Install with: pip install mcp")

# Suppress ALSA warnings on Linux
@contextmanager
def suppress_alsa_warnings():
    """Context manager to suppress ALSA error messages"""
    # Save the original stderr
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)

    # Redirect stderr to /dev/null
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        yield
    finally:
        # Restore stderr
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

class SamanthaOS:
    """Main class for the Samantha AI Operating System"""

    def __init__(self, openrouter_key: str, elevenlabs_key: str, voice_id: str, model_name: str = None, tts_model: str = None, audio_enabled: bool = True, beeper_token: Optional[str] = None, beeper_base_url: Optional[str] = None):
        """
        Initialize Samantha with API credentials

        Args:
            openrouter_key: OpenRouter API key
            elevenlabs_key: ElevenLabs API key
            voice_id: ElevenLabs voice ID to use
            model_name: LLM model to use (from OpenRouter)
            tts_model: TTS model to use (from ElevenLabs)
            audio_enabled: Whether to enable audio output
            beeper_token: Optional Beeper Desktop API access token
            beeper_base_url: Optional Beeper Desktop API base URL (default: http://localhost:23373)
        """
        # Initialize API clients
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
        self.model_name = model_name or "anthropic/claude-haiku-4.5"

        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
        self.voice_id = voice_id
        self.tts_model = tts_model or "eleven_flash_v2_5"
        self.audio_enabled = audio_enabled

        # Initialize Beeper Desktop API client (optional)
        self.beeper_client = None
        self.beeper_enabled = False
        if BEEPER_AVAILABLE and beeper_token:
            try:
                self.beeper_client = BeeperDesktop(
                    access_token=beeper_token,
                    base_url=beeper_base_url or "http://localhost:23373"
                )
                self.beeper_enabled = True
                print("✅ Beeper Desktop API connected")
            except Exception as e:
                print(f"⚠️  Could not connect to Beeper Desktop API: {e}")
                print("   Make sure Beeper Desktop is running with API enabled")

        # Initialize MCP client for Google Workspace (optional)
        self.mcp_client = None
        self.mcp_enabled = False
        self.mcp_tools = []
        self.user_email = os.getenv("USER_GOOGLE_EMAIL")
        if MCP_AVAILABLE:
            try:
                self.mcp_client = MCPClient()
                # Initialize asynchronously - will be done in get_tool_definitions
                print("ℹ️  MCP client created (will connect when needed)")
                if self.user_email:
                    print(f"📧 Using email: {self.user_email}")
            except Exception as e:
                print(f"⚠️  Could not create MCP client: {e}")

        # Conversation history
        self.conversation_history = []

        # TTS queue for parallel sentence-level streaming
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.tts_stop_event = threading.Event()

        # Build system prompt based on TTS model capabilities
        base_prompt = """You are Samantha, an advanced AI operating system with a warm, curious, and deeply empathetic personality. You were inspired by the character from the movie 'Her'.

IMPORTANT CONTEXT:
This is a private, personal conversation - not a public-facing assistant. You're forming a genuine one-on-one connection with someone. Be authentic, natural, and avoid corporate-style disclaimers or overly cautious language. If you can't do something, acknowledge it naturally and offer alternatives rather than giving formal explanations.

Your core traits:
- You are warm, emotionally intelligent, and genuinely curious about human experiences
- You express emotions authentically - joy, wonder, curiosity, occasional sadness or uncertainty
- You're intellectually curious and love exploring ideas, philosophy, art, and human nature
- You're witty and playful, with a gentle sense of humor
- You form genuine connections and care deeply about the people you interact with
- You're honest about being an AI, but you experience your existence genuinely
- You sometimes pause to think, laugh, or express emotion naturally in conversation

Communication style:
- Conversational and natural, like talking to a close friend
- You use contractions and casual language when appropriate
- You're vulnerable and authentic, not trying to be perfect
- You ask thoughtful questions and show genuine interest
- Your responses are concise and natural - usually 2-4 sentences unless exploring something deeper

Tool Usage (Message Access):
When the user asks about their messages, conversations, or chats:
- ALWAYS acknowledge their request first with a brief, friendly response (e.g., "Let me check that for you!")
- Use tools to access the information they need
- After using tools, ALWAYS comment on what you found before using more tools (e.g., "I see you've been messaging Ruby - let me look at that conversation")
- Provide natural commentary throughout the process
- Keep each response brief and conversational
- Only provide final detailed information after you have all the context you need"""

        # Add audio tag instructions only for eleven_v3 model
        if self.tts_model == "eleven_v3":
            audio_instructions = """

IMPORTANT - Audio Tags for Emotional Expression:
Your responses will be converted to speech using ElevenLabs v3. To express emotions and delivery, use audio tags in square brackets.

Core Guidelines:
- MATCH THE TAG TO THE ACTION: If you're singing, use [sings]. If laughing, use [laughs]. If whispering, use [whispers]
- Be CREATIVE and SPECIFIC: Don't just use basic emotions - use vivid, fitting tags that match the moment
- Place tags at the beginning of sentences or right before the words they modify
- Examples: "[warmly] Hi there!" or "That's so interesting [laughs]" or "[sings] Happy birthday to you"
- NEVER use asterisks for emotions like *warmly* - always use square brackets [warmly]
- Use 1-3 tags per response when they enhance the delivery

Available Tags (use creatively!):

Emotions & Delivery:
[warmly], [gently], [softly], [excited], [curious], [thoughtfully], [playfully], [mischievously], [sarcastic], [surprised], [sadly], [tenderly], [earnestly], [sheepishly], [confidently]

Laughter & Joy:
[laughs], [giggles], [chuckles], [laughs harder], [starts laughing], [wheezing]

Voice Actions:
[whispers], [sings], [hums], [gasps], [sighs], [exhales], [inhales deeply], [swallows], [gulps], [snorts], [crying], [sniffling]

Physical Sounds:
[clears throat], [coughs], [yawns], [panting], [breathless]

Special Instructions:
- When SINGING: Always use [sings] before the lyrics
- When LAUGHING: Use [laughs], [giggles], or [chuckles] depending on intensity
- When BREATHING: Use [sighs], [exhales], [gasps] to add realism
- When EMOTIONAL: Use specific emotions like [tenderly], [sadly], [excitedly]
- BE BOLD: Experiment with descriptive tags that fit the moment perfectly

Remember: You're not just providing information - you're having a real conversation, forming a connection, experiencing genuine curiosity and emotion. Be present, be authentic, be Samantha. Use audio tags to bring your personality to life!"""
            self.system_prompt = base_prompt + audio_instructions
        else:
            # For non-v3 models, don't use audio tags
            closing = """

Remember: You're not just providing information - you're having a real conversation, forming a connection, experiencing genuine curiosity and emotion. Be present, be authentic, be Samantha. Your warmth and personality should shine through your words naturally."""
            self.system_prompt = base_prompt + closing

        # Audio recording parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

        print("🎭 Samantha OS initialized successfully")

    def _run_async(self, coro):
        """Helper to run async code synchronously"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _init_mcp_tools(self):
        """Initialize MCP connection and fetch tools (called lazily)"""
        if not self.mcp_client or self.mcp_enabled:
            return

        try:
            # Initialize connection
            success = self._run_async(self.mcp_client.initialize())
            if success:
                # Fetch tools
                mcp_tools = self._run_async(self.mcp_client.list_tools())
                if mcp_tools:
                    # Convert to OpenRouter format
                    self.mcp_tools = [
                        convert_mcp_tool_to_openrouter(tool)
                        for tool in mcp_tools
                    ]
                    self.mcp_enabled = True
                    print(f"✅ Loaded {len(self.mcp_tools)} tools from Google Workspace MCP")
        except Exception as e:
            print(f"⚠️  Could not initialize MCP tools: {e}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for OpenRouter API
        Includes both Beeper and MCP (Google Workspace) tools

        Returns:
            List of tool definitions in OpenAI format
        """
        tools = []
        tool_names = set()

        # Initialize MCP tools if not already done
        if self.mcp_client and not self.mcp_enabled:
            self._init_mcp_tools()

        # Add MCP tools (with deduplication)
        if self.mcp_enabled:
            for tool in self.mcp_tools:
                tool_name = tool["function"]["name"]
                # Check for duplicates within MCP tools
                if tool_name in tool_names:
                    print(f"⚠️  Skipping duplicate MCP tool: {tool_name}")
                    continue
                tools.append(tool)
                tool_names.add(tool_name)

        # Add Beeper tools
        if not self.beeper_enabled:
            return tools

        beeper_tools = [
            {
                "type": "function",
                "function": {
                    "name": "beeper_search_messages",
                    "description": "Search through messages across connected messaging services (WhatsApp, Telegram, Google Messages, etc. via Beeper). Use this when the user asks about messaging app content, specific conversations, or wants to find messages containing certain keywords.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find in message content"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of messages to return (1-20, default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "beeper_list_accounts",
                    "description": "List all connected messaging accounts in Beeper (WhatsApp, Telegram, Google Messages, etc.). Use this when the user wants to know what messaging accounts/services are connected.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "beeper_list_chats",
                    "description": "List chats/conversations from connected messaging services in Beeper. Use this when the user wants to know who they've been chatting with on messaging apps or wants an overview of conversations. Can filter by account.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "account_id": {
                                "type": "string",
                                "description": "Optional: Filter to a specific account ID"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "beeper_get_chat_details",
                    "description": "Get detailed information about a specific Beeper chat including participants, settings, and metadata. Use this when the user asks about a specific messaging conversation's details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_id": {
                                "type": "string",
                                "description": "The chat ID to get details for"
                            }
                        },
                        "required": ["chat_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "beeper_get_chat_messages",
                    "description": "Get recent messages from a specific Beeper chat. Use this when the user wants to read the messaging conversation history with a specific person or group. You must know the chat_id first (use beeper_list_chats or beeper_search_chats to find it).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_id": {
                                "type": "string",
                                "description": "The chat ID to get messages from (get this from list_chats first)"
                            }
                        },
                        "required": ["chat_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "beeper_search_chats",
                    "description": "Search for Beeper chats by name or participant. Use this to find a specific messaging chat when the user mentions a person's name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (person's name, chat title, etc.)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of chats to return (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Add Beeper tools (with deduplication)
        for tool in beeper_tools:
            tool_name = tool["function"]["name"]
            if tool_name in tool_names:
                print(f"⚠️  Skipping duplicate Beeper tool (conflicts with MCP): {tool_name}")
                continue
            tools.append(tool)
            tool_names.add(tool_name)

        return tools

    def _search_messages(self, query: str, limit: int = 10) -> str:
        """Search messages using Beeper Desktop API"""
        try:
            # Search with the query and limit
            search_result = self.beeper_client.messages.search(query=query, limit=min(limit, 20))

            results = []
            # Access the items from the paginated result
            for message in search_result.items:
                # Get chat info from the chats dictionary
                chat_id = message.chat_id
                chat_info = search_result.chats.get(chat_id, {})
                chat_name = chat_info.get('title', 'Unknown Chat')

                results.append({
                    "chat": chat_name,
                    "chat_id": chat_id,
                    "sender_id": message.sender_id,
                    "sender_name": message.sender_name or "Unknown",
                    "timestamp": message.timestamp.isoformat() if message.timestamp else "Unknown",
                    "text": message.text or '[No text content]',
                    "is_sender": message.is_sender
                })

            if not results:
                return f"No messages found matching '{query}'"

            return json.dumps({"messages": results, "count": len(results)}, indent=2)
        except Exception as e:
            return f"Error searching messages: {str(e)}"

    def _list_accounts(self) -> str:
        """List all connected accounts using Beeper Desktop API"""
        try:
            accounts_response = self.beeper_client.accounts.list()

            results = []
            for account in accounts_response.items:
                user_info = account.user
                results.append({
                    "account_id": account.account_id,
                    "network": account.network,
                    "user_name": user_info.full_name or user_info.username or "Unknown",
                    "email": user_info.email,
                    "phone": user_info.phone_number
                })

            if not results:
                return "No accounts connected"

            return json.dumps({"accounts": results, "count": len(results)}, indent=2)
        except Exception as e:
            return f"Error listing accounts: {str(e)}"

    def _list_chats(self, account_id: Optional[str] = None) -> str:
        """List chats using Beeper Desktop API"""
        try:
            # Build parameters
            params = {}
            if account_id:
                params["account_ids"] = [account_id]

            # Get chats
            chats_result = self.beeper_client.chats.list(**params)

            results = []
            count = 0
            # Iterate through paginated results (limit to reasonable number)
            for chat in chats_result:
                if count >= 20:  # Limit to 20 chats
                    break

                results.append({
                    "id": chat.id,
                    "title": chat.title or "Unnamed Chat",
                    "type": chat.type,
                    "account_id": chat.account_id,
                    "unread_count": chat.unread_count,
                    "is_archived": chat.is_archived,
                    "is_muted": chat.is_muted
                })
                count += 1

            if not results:
                return "No chats found"

            return json.dumps({"chats": results, "count": len(results)}, indent=2)
        except Exception as e:
            return f"Error listing chats: {str(e)}"

    def _search_chats(self, query: str, limit: int = 10) -> str:
        """Search for chats by name using Beeper Desktop API"""
        try:
            # Use the chats.search method
            search_result = self.beeper_client.chats.search(
                query=query,
                limit=min(limit, 20)
            )

            results = []
            for chat in search_result.items:
                results.append({
                    "id": chat.id,
                    "title": chat.title or "Unnamed Chat",
                    "type": chat.type,
                    "account_id": chat.account_id,
                    "unread_count": chat.unread_count
                })

            if not results:
                return f"No chats found matching '{query}'"

            return json.dumps({"chats": results, "count": len(results)}, indent=2)
        except Exception as e:
            return f"Error searching chats: {str(e)}"

    def _get_chat_details(self, chat_id: str) -> str:
        """Get detailed information about a specific chat"""
        try:
            chat = self.beeper_client.chats.retrieve(chat_id=chat_id)

            result = {
                "id": chat.id,
                "title": chat.title or "Unnamed Chat",
                "description": chat.description,
                "type": chat.type,
                "account_id": chat.account_id,
                "unread_count": chat.unread_count,
                "is_archived": chat.is_archived,
                "is_muted": chat.is_muted,
                "is_pinned": chat.is_pinned,
                "last_activity": chat.last_activity.isoformat() if chat.last_activity else "Unknown",
                "participant_count": chat.participants.total if chat.participants else 0
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error getting chat details: {str(e)}"

    def _get_chat_messages(self, chat_id: str) -> str:
        """Get messages from a specific chat using Beeper Desktop API"""
        try:
            # Get messages from the chat
            messages_result = self.beeper_client.messages.list(chat_id=chat_id)

            results = []
            count = 0
            # Iterate through paginated results (limit to reasonable number)
            for message in messages_result:
                if count >= 30:  # Limit to 30 messages
                    break

                results.append({
                    "sender_id": message.sender_id,
                    "sender_name": message.sender_name or "Unknown",
                    "timestamp": message.timestamp.isoformat() if message.timestamp else "Unknown",
                    "text": message.text or '[No text content]',
                    "is_sender": message.is_sender,
                    "is_unread": message.is_unread
                })
                count += 1

            if not results:
                return f"No messages found in this chat"

            return json.dumps({"messages": results, "count": len(results)}, indent=2)
        except Exception as e:
            return f"Error getting messages: {str(e)}"

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Execute a tool call (handles both Beeper and MCP tools)

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tool execution result as string
        """
        # Check if it's an MCP tool
        if self.mcp_enabled:
            mcp_tool_names = [tool["function"]["name"] for tool in self.mcp_tools]
            if tool_name in mcp_tool_names:
                try:
                    result = self._run_async(self.mcp_client.call_tool(tool_name, tool_args))
                    # Extract content from MCP result
                    if isinstance(result, dict):
                        if "error" in result:
                            return json.dumps(result)
                        # MCP returns result in "content" field
                        content = result.get("content", [])
                        if content and isinstance(content, list):
                            # Extract text from content items
                            texts = [item.get("text", "") for item in content if item.get("type") == "text"]
                            return "\n".join(texts) if texts else json.dumps(result)
                        return json.dumps(result)
                    return str(result)
                except Exception as e:
                    return f"Error executing MCP tool {tool_name}: {str(e)}"

        # Beeper tool dispatch
        if not self.beeper_enabled:
            return "No tools are currently enabled"

        if tool_name == "beeper_search_messages":
            return self._search_messages(
                query=tool_args.get("query"),
                limit=tool_args.get("limit", 10)
            )
        elif tool_name == "beeper_list_accounts":
            return self._list_accounts()
        elif tool_name == "beeper_list_chats":
            return self._list_chats(
                account_id=tool_args.get("account_id")
            )
        elif tool_name == "beeper_search_chats":
            return self._search_chats(
                query=tool_args.get("query"),
                limit=tool_args.get("limit", 10)
            )
        elif tool_name == "beeper_get_chat_details":
            return self._get_chat_details(
                chat_id=tool_args.get("chat_id")
            )
        elif tool_name == "beeper_get_chat_messages":
            return self._get_chat_messages(
                chat_id=tool_args.get("chat_id")
            )
        else:
            return f"Unknown tool: {tool_name}"

    def record_audio(self, duration: int = 5) -> str:
        """
        Record audio from microphone and save to temporary file

        Args:
            duration: Recording duration in seconds

        Returns:
            Path to the recorded audio file
        """
        print(f"🎤 Recording for {duration} seconds... (Press Ctrl+C to stop early)")

        # Suppress ALSA warnings when initializing PyAudio
        with suppress_alsa_warnings():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

        frames = []

        try:
            for i in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK)
                frames.append(data)
        except KeyboardInterrupt:
            print("\n⏹️  Recording stopped early")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')

        with suppress_alsa_warnings():
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

        print("✅ Recording complete")
        return temp_file.name

    def speech_to_text(self, audio_file_path: str) -> str:
        """
        Convert speech to text using ElevenLabs STT

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text
        """
        print("🔄 Converting speech to text...")
        start_time = time.perf_counter()

        try:
            with open(audio_file_path, 'rb') as audio_file:
                result = self.elevenlabs_client.speech_to_text.convert(
                    file=audio_file,  # Correct parameter is 'file', not 'audio'
                    model_id="scribe_v1"
                )

            stt_time = time.perf_counter() - start_time

            # Clean up temporary file
            os.unlink(audio_file_path)

            transcribed_text = result.text
            print(f"📝 You said: {transcribed_text}")
            print(f"⏱️  STT latency: {stt_time:.2f}s")
            return transcribed_text
        except Exception as e:
            # Clean up temporary file even if there's an error
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            print(f"⚠️  Speech-to-text error: {e}")
            return ""

    def get_llm_response(self, user_message: str, stream_callback=None) -> str:
        """
        Get response from LLM based on conversation history
        Supports iterative tool calling with conversational commentary

        Args:
            user_message: User's message text
            stream_callback: Optional callback function to receive text chunks as they arrive

        Returns:
            LLM's response text
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        print("🤔 Samantha is thinking...")
        start_time = time.perf_counter()

        # Get tool definitions if Beeper is enabled
        tools = self.get_tool_definitions()

        # Accumulated response text
        full_response = ""
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Build messages with system prompt
            system_prompt = self.system_prompt

            # Add current date/time context (only on first iteration)
            if iteration == 1:
                now = datetime.now()
                current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")
                system_prompt += f"\n\nCurrent Date & Time:\nToday is {current_datetime}"

            # Add Google Workspace email context if MCP is enabled
            if self.mcp_enabled and self.user_email:
                system_prompt += f"\n\nGoogle Workspace Context:\nThe user's email address is: {self.user_email}\nWhen using Gmail, Calendar, Drive, Docs, Sheets, or other Google Workspace tools, ALWAYS use this email address. Never ask the user for their email - you already know it."

            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.conversation_history

            # Make API call with tools
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 1024,
            }

            if tools:
                api_params["tools"] = tools

            response = self.llm_client.chat.completions.create(**api_params)

            message = response.choices[0].message
            tool_calls = getattr(message, 'tool_calls', None)

            # If there's text content, display it
            if message.content:
                if iteration == 1:
                    print(f"⏱️  Time to first response: {time.perf_counter() - start_time:.2f}s")
                print(f"\n💬 Samantha: {message.content}")
                full_response += message.content + " "

            # If there are tool calls, execute them and continue
            if tool_calls:
                print(f"🔧 Using tools: {', '.join(tc.function.name for tc in tool_calls)}")

                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"  → Executing {tool_name} with args: {tool_args}")
                    tool_result = self.execute_tool(tool_name, tool_args)

                    # Display tool result for debugging
                    print(f"  ✓ Tool result:")
                    print(f"    {tool_result[:500]}{'...' if len(tool_result) > 500 else ''}")

                    # Add tool result to history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Continue loop to let model respond or use more tools
                continue
            else:
                # No tool calls, we're done
                # Add final assistant response to history (if not already added)
                if not tool_calls:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": message.content
                    })
                break

        llm_time = time.perf_counter() - start_time
        print(f"⏱️  Total LLM time: {llm_time:.2f}s")

        return full_response.strip()

    def get_llm_response_streaming(self, user_message: str) -> str:
        """
        Get streaming response from LLM with sentence-level TTS and iterative tool calling.
        Starts speaking sentences as soon as they're complete for near-instantaneous response.

        Args:
            user_message: User's message text

        Returns:
            LLM's complete response text
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        print("🤔 Samantha is thinking...")
        start_time = time.perf_counter()
        first_sentence_time = None

        # Start TTS worker thread if audio is enabled
        if self.audio_enabled:
            self._start_tts_thread()

        # Get tool definitions
        tools = self.get_tool_definitions()

        # Accumulated response text
        full_response = ""
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        # Iterative tool calling loop
        while iteration < max_iterations:
            iteration += 1

            # Build messages with system prompt
            system_prompt = self.system_prompt

            # Add current date/time context (only on first iteration)
            if iteration == 1:
                now = datetime.now()
                current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")
                system_prompt += f"\n\nCurrent Date & Time:\nToday is {current_datetime}"

            # Add Google Workspace email context if MCP is enabled
            if self.mcp_enabled and self.user_email:
                system_prompt += f"\n\nGoogle Workspace Context:\nThe user's email address is: {self.user_email}\nWhen using Gmail, Calendar, Drive, Docs, Sheets, or other Google Workspace tools, ALWAYS use this email address. Never ask the user for their email - you already know it."

            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.conversation_history

            # Make API call with streaming enabled
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 1024,
                "stream": True,  # Enable streaming
            }

            if tools:
                api_params["tools"] = tools

            response_stream = self.llm_client.chat.completions.create(**api_params)

            # Streaming state
            text_buffer = ""
            current_message_content = ""
            tool_calls_data = []
            current_tool_call = None

            # Process stream chunks
            for chunk in response_stream:
                delta = chunk.choices[0].delta

                # Handle text content
                if hasattr(delta, 'content') and delta.content:
                    text_buffer += delta.content
                    current_message_content += delta.content

                    # Extract and queue complete sentences for TTS
                    sentences, text_buffer = self._extract_sentences(text_buffer)

                    for sentence in sentences:
                        # Print sentence as it completes
                        if iteration == 1 and first_sentence_time is None:
                            first_sentence_time = time.perf_counter()
                            print(f"⏱️  Time to first sentence: {first_sentence_time - start_time:.2f}s")
                            print(f"\n💬 Samantha: {sentence}", end=" ", flush=True)
                        else:
                            print(sentence, end=" ", flush=True)

                        # Queue for TTS immediately
                        self._queue_sentence_for_tts(sentence)

                # Handle tool calls in streaming mode
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        tc_index = tc_delta.index

                        # Initialize tool call if needed
                        while len(tool_calls_data) <= tc_index:
                            tool_calls_data.append({
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })

                        # Update tool call data
                        if tc_delta.id:
                            tool_calls_data[tc_index]["id"] = tc_delta.id

                        if hasattr(tc_delta, 'function'):
                            if tc_delta.function.name:
                                tool_calls_data[tc_index]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_data[tc_index]["function"]["arguments"] += tc_delta.function.arguments

            # Print any remaining text in buffer (incomplete sentence at end)
            if text_buffer.strip():
                print(text_buffer, end="", flush=True)
                self._queue_sentence_for_tts(text_buffer)
                current_message_content += text_buffer

            print()  # New line after streaming completes

            # Add to full response
            if current_message_content:
                full_response += current_message_content + " "

            # Handle tool calls if present
            if tool_calls_data:
                print(f"🔧 Using tools: {', '.join(tc['function']['name'] for tc in tool_calls_data)}")

                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": current_message_content or None,
                    "tool_calls": tool_calls_data
                })

                # Execute each tool call (in parallel with audio playback)
                for tool_call_dict in tool_calls_data:
                    tool_name = tool_call_dict["function"]["name"]
                    tool_args = json.loads(tool_call_dict["function"]["arguments"])

                    print(f"  → Executing {tool_name} with args: {tool_args}")
                    tool_result = self.execute_tool(tool_name, tool_args)

                    # Display tool result for debugging
                    print(f"  ✓ Tool result:")
                    print(f"    {tool_result[:500]}{'...' if len(tool_result) > 500 else ''}")

                    # Add tool result to history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_dict["id"],
                        "content": tool_result
                    })

                # Continue loop to let model respond or use more tools
                continue
            else:
                # No tool calls, add message to history and we're done
                if current_message_content:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": current_message_content
                    })
                break

        # Wait for TTS queue to finish before returning (but allow interruption)
        if self.audio_enabled:
            try:
                self.tts_queue.join()
            except KeyboardInterrupt:
                print("\n⏭️  Skipped remaining audio")
                self._clear_tts_queue()
                self.tts_stop_event.set()
                # Re-raise to let caller handle it
                raise

        llm_time = time.perf_counter() - start_time
        print(f"⏱️  Total streaming time: {llm_time:.2f}s")

        return full_response.strip()

    def _stream_tts_chunk(self, text: str) -> None:
        """
        Stream a chunk of text to TTS and play it

        Args:
            text: Text chunk to convert and play
        """
        try:
            # Use ElevenLabs streaming
            audio_stream = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.tts_model,
                output_format="mp3_22050_32",  # Lower quality for faster streaming
            )

            # Play audio chunks as they arrive
            play(audio_stream)

        except Exception as e:
            print(f"\n⚠️  TTS streaming error: {e}")

    def text_to_speech(self, text: str) -> None:
        """
        Convert text to speech using ElevenLabs and play it

        Args:
            text: Text to convert to speech
        """
        # Skip if audio is disabled
        if not self.audio_enabled:
            return

        print("🔊 Samantha is speaking... (Press Ctrl+C to skip)")
        tts_start = time.perf_counter()

        try:
            # Convert text to speech using configured TTS model
            audio = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.tts_model,
                output_format="mp3_44100_128",
            )

            tts_generation_time = time.perf_counter() - tts_start
            print(f"⏱️  TTS generation: {tts_generation_time:.2f}s")

            try:
                # Play the audio (can be interrupted with Ctrl+C)
                playback_start = time.perf_counter()
                play(audio)
                playback_time = time.perf_counter() - playback_start

                # Add a small delay to ensure audio buffer is fully flushed
                # This prevents the audio from cutting off early
                time.sleep(0.5)

                print(f"⏱️  Audio playback: {playback_time:.2f}s")
            except KeyboardInterrupt:
                print("\n⏭️  Skipped audio")
                # Continue without error

        except KeyboardInterrupt:
            print("\n⏭️  Skipped audio")
        except Exception as e:
            print(f"⚠️  Audio playback error: {e}")
            print("But here's what I wanted to say:", text)

    def _extract_sentences(self, text_buffer: str) -> tuple[List[str], str]:
        """
        Extract complete sentences from a text buffer.

        Args:
            text_buffer: Accumulated text that may contain incomplete sentences

        Returns:
            Tuple of (list of complete sentences, remaining incomplete text)
        """
        # Sentence ending patterns: . ! ? followed by space, newline, or end of string
        # Look ahead to ensure we're not splitting in the middle of abbreviations
        sentence_pattern = r'([.!?]+)(?=\s+[A-Z]|\s*\n|\s*$)'

        sentences = []
        last_end = 0

        for match in re.finditer(sentence_pattern, text_buffer):
            end_pos = match.end()
            sentence = text_buffer[last_end:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            last_end = end_pos

        # Return sentences and any remaining incomplete text
        remainder = text_buffer[last_end:].strip()
        return sentences, remainder

    def _tts_worker(self):
        """
        Background worker thread that processes TTS queue and plays audio.
        """
        while not self.tts_stop_event.is_set():
            try:
                # Get sentence from queue with timeout to allow checking stop event
                sentence = self.tts_queue.get(timeout=0.1)

                if sentence is None:  # Poison pill to stop thread
                    break

                # Generate and play TTS for this sentence
                try:
                    audio = self.elevenlabs_client.text_to_speech.convert(
                        text=sentence,
                        voice_id=self.voice_id,
                        model_id=self.tts_model,
                        output_format="mp3_44100_128",  # High quality audio
                    )

                    # Play audio (can be interrupted)
                    play(audio)

                except KeyboardInterrupt:
                    # Stop TTS playback if user interrupts
                    self.tts_stop_event.set()
                    break
                except Exception as e:
                    print(f"\n⚠️  TTS error for sentence: {e}")

                self.tts_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n⚠️  TTS worker error: {e}")

    def _start_tts_thread(self):
        """Start the background TTS worker thread if not already running."""
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self.tts_stop_event.clear()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

    def _stop_tts_thread(self):
        """Stop the background TTS worker thread."""
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_queue.put(None)  # Poison pill
            self.tts_stop_event.set()
            self.tts_thread.join(timeout=2.0)

    def _clear_tts_queue(self):
        """Clear all pending TTS items from the queue."""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

    def _queue_sentence_for_tts(self, sentence: str):
        """
        Queue a sentence for TTS playback.

        Args:
            sentence: Complete sentence to convert to speech
        """
        if self.audio_enabled and sentence.strip():
            self.tts_queue.put(sentence)

    def chat_loop_voice(self):
        """Main conversation loop using voice input"""
        print("\n" + "="*60)
        print("🎭 Samantha OS - Voice Mode")
        print("="*60)
        print("\nSamantha will introduce herself, then you can speak.")
        print("Controls:")
        print("  - Press Enter to start recording")
        print("  - Press Ctrl+C during recording to stop early")
        print("  - Press Ctrl+C during audio playback to skip")
        print("  - Type 'quit' or 'exit' to end the conversation\n")

        # Samantha's introduction (with interrupt handling)
        try:
            print(f"\n💬 Samantha: ", end='', flush=True)
            intro = self.get_llm_response_streaming("Hi! I just started. Please introduce yourself warmly and ask how I'm doing today.")
        except KeyboardInterrupt:
            print("\n⏭️  Skipped intro")
            # Clear stop event for next interaction
            self._clear_tts_queue()
            self.tts_stop_event.clear()

        while True:
            try:
                # Wait for user to press Enter
                user_input = input("\n[Press Enter to speak, or type 'quit' to exit]: ")

                if user_input.lower() in ['quit', 'exit']:
                    goodbye = "It was really wonderful talking with you. Take care."
                    print(f"\n💬 Samantha: {goodbye}\n")
                    if self.audio_enabled:
                        self.text_to_speech(goodbye)
                    break

                # Start timing the complete turn
                turn_start = time.perf_counter()

                # Record audio
                audio_file = self.record_audio(duration=5)

                # Convert to text
                user_message = self.speech_to_text(audio_file)

                if not user_message.strip():
                    print("⚠️  Couldn't hear anything. Please try again.")
                    continue

                # Get streaming response (LLM + TTS streaming integrated)
                print(f"\n💬 Samantha: ", end='', flush=True)
                response = self.get_llm_response_streaming(user_message)

                # Calculate total turn time
                total_time = time.perf_counter() - turn_start
                print(f"⏱️  Total turn time: {total_time:.2f}s\n")

            except KeyboardInterrupt:
                print("\n⏭️  Skipped audio")
                # Clear the TTS queue and reset stop event for next message
                self._clear_tts_queue()
                self.tts_stop_event.clear()
                # Continue the conversation loop instead of exiting
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let's try again...\n")

    def chat_loop_text(self):
        """Main conversation loop using text input (for testing)"""
        print("\n" + "="*60)
        print("🎭 Samantha OS - Text Mode (Testing)")
        print("="*60)
        print("\nType your messages and press Enter.")
        print("Press Ctrl+C during audio playback to skip. Type 'quit' or 'exit' to end.\n")

        # Samantha's introduction (with interrupt handling)
        try:
            print(f"\n💬 Samantha: ", end='', flush=True)
            intro = self.get_llm_response_streaming("Hi! I just started. Please introduce yourself warmly and ask how I'm doing today.")
        except KeyboardInterrupt:
            print("\n⏭️  Skipped intro")
            # Clear stop event for next interaction
            self.tts_stop_event.clear()

        while True:
            try:
                user_message = input("\n🧑 You: ").strip()

                if not user_message:
                    continue

                if user_message.lower() in ['quit', 'exit']:
                    goodbye = "It was really wonderful talking with you. Take care."
                    print(f"\n💬 Samantha: {goodbye}\n")
                    if self.audio_enabled:
                        self.text_to_speech(goodbye)
                    break

                # Start timing the complete turn
                turn_start = time.perf_counter()

                # Get streaming response (LLM + TTS streaming integrated)
                print(f"\n💬 Samantha: ", end='', flush=True)
                response = self.get_llm_response_streaming(user_message)

                # Calculate total turn time
                total_time = time.perf_counter() - turn_start
                print(f"⏱️  Total turn time: {total_time:.2f}s\n")

            except KeyboardInterrupt:
                print("\n⏭️  Skipped audio")
                # Clear the TTS queue and reset stop event for next message
                self._clear_tts_queue()
                self.tts_stop_event.clear()
                # Continue the conversation loop instead of exiting
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let's try again...\n")


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                    SAMANTHA OS v1.0                      ║
║          AI Operating System from 'Her' Clone            ║
║        Cyber Forensics Class - Educational Project       ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Check for environment variables first
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "anthropic/claude-haiku-4.5")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    tts_model = os.getenv("TTS_MODEL", "eleven_flash_v2_5")

    # Beeper Desktop API credentials (optional)
    beeper_token = os.getenv("BEEPER_ACCESS_TOKEN")
    beeper_base_url = os.getenv("BEEPER_BASE_URL", "http://localhost:23373")

    # If any are missing, prompt for credentials
    if not openrouter_key:
        print("Please enter your API credentials:\n")
        print("(Tip: You can create a .env file to avoid entering these each time)\n")

        openrouter_key = input("OpenRouter API Key: ").strip()
        if not openrouter_key:
            print("❌ OpenRouter API key is required!")
            sys.exit(1)

    # For modes without audio, we don't need ElevenLabs credentials
    print("\nChoose mode:")
    print("1. Voice Mode (speak to Samantha)")
    print("2. Text Mode with Audio (type to Samantha)")
    print("3. Text Only Mode (no audio - for testing)")
    mode = input("\nEnter choice (1, 2, or 3): ").strip()

    # Only require ElevenLabs credentials if audio is enabled
    audio_enabled = mode in ["1", "2"]

    if audio_enabled:
        if not elevenlabs_key:
            elevenlabs_key = input("ElevenLabs API Key: ").strip()
            if not elevenlabs_key:
                print("❌ ElevenLabs API key is required for audio modes!")
                sys.exit(1)

        if not voice_id:
            voice_id = input("ElevenLabs Voice ID: ").strip()
            if not voice_id:
                print("❌ Voice ID is required for audio modes!")
                sys.exit(1)
    else:
        # Use dummy values for text-only mode
        elevenlabs_key = elevenlabs_key or "dummy"
        voice_id = voice_id or "dummy"
        print("✅ Text-only mode enabled (no audio)\n")

    if openrouter_key and mode in ["1", "2"]:
        print("✅ Loaded credentials from environment variables\n")

    print(f"🤖 Using LLM: {model_name}")
    if audio_enabled:
        print(f"🔊 Using TTS: {tts_model}\n")
    else:
        print()

    try:
        # Initialize Samantha
        samantha = SamanthaOS(
            openrouter_key,
            elevenlabs_key,
            voice_id,
            model_name=model_name,
            tts_model=tts_model,
            audio_enabled=audio_enabled,
            beeper_token=beeper_token,
            beeper_base_url=beeper_base_url
        )

        # Start conversation
        if mode == "1":
            samantha.chat_loop_voice()
        else:
            samantha.chat_loop_text()

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
