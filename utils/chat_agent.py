"""Chat agent for player interaction in Gradient Bang."""

import json
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from utils.base_llm_agent import BaseLLMAgent, LLMConfig
from utils.api_client import AsyncGameClient
from utils.prompts import GAME_DESCRIPTION, CHAT_INSTRUCTIONS


def create_chat_system_prompt() -> str:
    """Create the system prompt for the chat agent."""
    return f"""{GAME_DESCRIPTION}

{CHAT_INSTRUCTIONS}"""


def get_chat_tool_definitions() -> List[Dict[str, Any]]:
    """Get tool definitions for the chat agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move to an adjacent sector (single jump only)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_sector": {
                            "type": "integer",
                            "description": "The adjacent sector to move to",
                        }
                    },
                    "required": ["to_sector"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_task",
                "description": "Start a complex multi-step task for navigation, trading, or exploration",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Natural language description of the task to execute",
                        },
                        "context": {
                            "type": "string",
                            "description": "Relevant conversation history or clarifications",
                        },
                    },
                    "required": ["task_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_task",
                "description": "Cancel the currently running task",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": "Get the ship's current status including location, cargo, and credits",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "view_map",
                "description": "Display the ship's accumulated map knowledge",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "show_ports": {
                            "type": "boolean",
                            "description": "Include port information in the output",
                            "default": True,
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scan_port",
                "description": "Scan a specific port for trading information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector": {
                            "type": "integer",
                            "description": "Sector number containing the port to scan",
                        }
                    },
                    "required": ["sector"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_cargo",
                "description": "Check the ship's current cargo hold",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reset_client",
                "description": "Reset the game client connection (use for error recovery)",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "force_refresh",
                "description": "Force refresh all cached data from the server",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


class ChatAgent(BaseLLMAgent):
    """Chat agent for player interaction with ship intelligence personality."""

    def __init__(
        self,
        config: LLMConfig,
        game_client: AsyncGameClient,
        tool_executor: "AsyncToolExecutor",
        task_callback: Optional[Callable[[str, str], asyncio.Task]] = None,
        cancel_task_callback: Optional[Callable[[], None]] = None,
        get_task_progress_callback: Optional[Callable[[], str]] = None,
        verbose_prompts: bool = False,
        output_callback: Optional[Callable[[str], None]] = None,
        debug_callback: Optional[Callable[[List[Dict[str, Any]], Optional[str]], None]] = None,
    ):
        """Initialize the chat agent.

        Args:
            config: LLM configuration
            game_client: Shared game client instance
            tool_executor: Shared tool executor that knows the character ID
            task_callback: Callback to start tasks (receives task description and context)
            cancel_task_callback: Callback to cancel current task
            get_task_progress_callback: Callback to get current task progress
            verbose_prompts: Whether to print messages as they're added
            output_callback: Optional callback for output lines
            debug_callback: Optional callback to send messages list for debugging
        """
        super().__init__(config, verbose_prompts, output_callback)
        self.game_client = game_client
        self.tool_executor = tool_executor
        self.task_callback = task_callback
        self.cancel_task_callback = cancel_task_callback
        self.get_task_progress_callback = get_task_progress_callback
        self.debug_callback = debug_callback
        self.current_task: Optional[asyncio.Task] = None
        self.system_prompt = create_chat_system_prompt()
        self.request_start_time: Optional[float] = None

    def initialize_conversation(self):
        """Initialize a new conversation with the system prompt."""
        self.clear_messages()
        system_message = {"role": "system", "content": self.system_prompt}
        self.add_message(system_message)

    async def process_message(
        self, user_input: str, task_progress: Optional[str] = None
    ) -> str:
        """Process a user message and return the assistant's response.

        Args:
            user_input: The user's message
            task_progress: Optional task progress to include

        Returns:
            The assistant's response text
        """
        if task_progress:
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<task_progress>\n{task_progress}\n</task_progress>",
                    },
                    {"type": "text", "text": user_input},
                ],
            }
        else:
            user_message = {"role": "user", "content": user_input}

        self.add_message(user_message)

        # Keep processing tool calls until the assistant responds with content
        max_tool_rounds = 10  # Prevent infinite loops
        total_start_time = time.time()
        
        for round_num in range(max_tool_rounds):
            # Send messages to debug callback before inference
            if self.debug_callback:
                self.debug_callback(self.messages.copy(), f"Request {round_num + 1} in progress...")
            
            # Track timing for this request
            self.request_start_time = time.time()
            
            assistant_message = await self.get_assistant_response(
                tools=get_chat_tool_definitions()
                # reasoning_effort not passed - gpt-4.1 doesn't support it
            )
            
            # Calculate response time for this round
            round_time = time.time() - self.request_start_time

            self.add_message(assistant_message)

            # If there are tool calls, execute them and continue the loop
            if "tool_calls" in assistant_message:
                tool_messages = await self.execute_chat_tools(
                    assistant_message["tool_calls"]
                )
                for tool_message in tool_messages:
                    self.add_message(tool_message)
                
                # Continue to next iteration to get follow-up response
                continue
            
            # No tool calls - we have the final response
            # Calculate total time
            total_time = time.time() - total_start_time
            
            # Update debug panel with final state and timing
            if self.debug_callback:
                if round_num == 0:
                    # Single round
                    self.debug_callback(self.messages.copy(), f"Complete ({total_time:.2f}s)")
                else:
                    # Multiple rounds
                    self.debug_callback(self.messages.copy(), f"Complete after {round_num + 1} rounds ({total_time:.2f}s)")
            
            return assistant_message.get("content", "")
        
        # If we hit max rounds, return an error message
        total_time = time.time() - total_start_time
        if self.debug_callback:
            self.debug_callback(self.messages.copy(), f"Max tool rounds exceeded ({total_time:.2f}s)")
        
        return "I apologize, but I've exceeded the maximum number of tool execution rounds. Please try simplifying your request."

    async def execute_chat_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute chat-specific tools.

        Args:
            tool_calls: List of tool calls from the assistant

        Returns:
            List of tool messages with results
        """
        tool_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            result = await self._execute_single_tool(tool_name, tool_args)

            tool_message = self.format_tool_message(tool_call["id"], result)
            tool_messages.append(tool_message)

        return tool_messages

    async def _execute_single_tool(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single chat tool.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tool execution result
        """
        try:
            # Map chat tool names to tool executor names where applicable
            tool_executor_mapping = {
                "move": "move",
                "get_status": "my_status",
                # view_map and scan_port need special handling, not direct delegation
            }

            # Delegate to tool executor for supported tools
            if tool_name in tool_executor_mapping:
                executor_tool_name = tool_executor_mapping[tool_name]
                return await self.tool_executor.execute_tool(
                    executor_tool_name, tool_args
                )

            # Handle chat-specific tools
            if tool_name == "start_task":
                if self.current_task and not self.current_task.done():
                    return {
                        "success": False,
                        "error": "A task is already running. Stop it first.",
                    }

                if self.task_callback:
                    task_desc = tool_args.get("task_description", "")
                    context = tool_args.get("context", "")

                    status = await self.game_client.my_status()
                    game_state = {
                        "current_sector": status.sector,
                        "credits": 1000,  # Mock until server provides
                        "cargo": {"fuel_ore": 0, "organics": 0, "equipment": 0},
                    }

                    self.current_task = self.task_callback(
                        f"{context}\n{task_desc}" if context else task_desc, game_state
                    )
                    return {"success": True, "message": "Task started"}
                else:
                    return {"success": False, "error": "Task execution not configured"}

            elif tool_name == "stop_task":
                if self.current_task and not self.current_task.done():
                    # Get any task progress before cancelling
                    task_progress = ""
                    if self.get_task_progress_callback:
                        task_progress = self.get_task_progress_callback()

                    # Use the cancel callback which properly sets the TaskAgent's cancelled flag
                    if self.cancel_task_callback:
                        self.cancel_task_callback()
                    else:
                        # Fallback to direct cancellation if no callback
                        self.current_task.cancel()

                    await self.game_client.my_status(force_refresh=True)

                    # Include task progress and instruction in the tool response
                    return {
                        "success": True,
                        "message": "Task cancelled",
                        "task_progress": task_progress,
                        "text_instruction": "The task was cancelled. Please acknowledge the cancellation and summarize what was done before stopping.",
                    }
                else:
                    return {"success": False, "error": "No task is currently running"}

            elif tool_name == "view_map":
                show_ports = tool_args.get("show_ports", True)
                map_data = await self.game_client.my_map(
                    self.tool_executor.character_id
                )

                sectors_visited = map_data.get("sectors_visited", {})
                sectors_with_ports = []
                if show_ports:
                    for sector_id, sector_data in sectors_visited.items():
                        if sector_data.get("port_info"):
                            port = sector_data["port_info"]
                            sectors_with_ports.append(
                                {
                                    "sector": sector_id,
                                    "class": port.get("class"),
                                    "buys": port.get("buys", []),
                                    "sells": port.get("sells", []),
                                }
                            )

                return {
                    "success": True,
                    "total_sectors_visited": map_data.get("total_sectors_visited", len(sectors_visited)),
                    "ports_discovered": len(sectors_with_ports),
                    "ports": sectors_with_ports if show_ports else None,
                }

            elif tool_name == "scan_port":
                sector = tool_args.get("sector")
                map_data = await self.game_client.my_map(
                    self.tool_executor.character_id
                )

                sectors_visited = map_data.get("sectors_visited", {})
                if str(sector) in sectors_visited:
                    sector_data = sectors_visited[str(sector)]
                    if sector_data.get("port_info"):
                        port = sector_data["port_info"]
                        return {
                            "success": True,
                            "sector": sector,
                            "port": {
                                "class": port.get("class"),
                                "code": port.get("code"),
                                "buys": port.get("buys", []),
                                "sells": port.get("sells", []),
                                "stock": port.get("stock", {}),
                                "demand": port.get("demand", {}),
                            },
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"No port in sector {sector}",
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Sector {sector} not yet visited",
                    }

            elif tool_name == "check_cargo":
                # Mock cargo data until server provides it
                return {
                    "success": True,
                    "cargo": {"fuel_ore": 0, "organics": 0, "equipment": 0},
                    "total": 0,
                }

            elif tool_name == "reset_client":
                await self.game_client.close()
                self.game_client = AsyncGameClient(
                    base_url=self.game_client.base_url,
                    character_id=self.tool_executor.character_id,
                )
                return {"success": True, "message": "Client connection reset"}

            elif tool_name == "force_refresh":
                await self.game_client.my_status(
                    self.tool_executor.character_id, force_refresh=True
                )
                await self.game_client.my_map(
                    self.tool_executor.character_id, force_refresh=True
                )
                return {"success": True, "message": "All cached data refreshed"}

            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def is_task_running(self) -> bool:
        """Check if a task is currently running."""
        return self.current_task is not None and not self.current_task.done()
