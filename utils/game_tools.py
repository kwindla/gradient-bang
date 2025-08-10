"""Tool definitions for LLM agents in Gradient Bang."""

from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
import asyncio
from utils.api_client import AsyncGameClient


class PlotCourseTool(BaseModel):
    """Calculate the shortest path between two sectors."""
    from_sector: int = Field(..., description="Starting sector ID")
    to_sector: int = Field(..., description="Destination sector ID")


class MoveTool(BaseModel):
    """Move to an adjacent sector."""
    to_sector: int = Field(..., description="Adjacent sector ID to move to")


class MyStatusTool(BaseModel):
    """Get current character status including position."""
    pass  # No parameters needed


class WaitForTimeTool(BaseModel):
    """Wait for a specified number of seconds."""
    seconds: float = Field(..., ge=0, le=60, description="Number of seconds to wait (max 60)")


class MyMapTool(BaseModel):
    """Get the character's map knowledge including visited sectors and known ports."""
    pass  # No parameters needed



class FindPortTool(BaseModel):
    """Find the nearest known port, optionally filtering by commodity type."""
    from_sector: Optional[int] = Field(
        None,
        description="Optional: Sector to search from (defaults to current sector)"
    )
    commodity: Optional[Literal["fuel_ore", "organics", "equipment"]] = Field(
        None, 
        description="Optional: The commodity to search for (must be exact: 'fuel_ore', 'organics', or 'equipment')"
    )
    buy_or_sell: Optional[Literal["buy", "sell"]] = Field(
        None,
        description="Optional: Whether to find a port that 'buy's or 'sell's the commodity (required if commodity is specified)"
    )


class FinishedTool(BaseModel):
    """Signal that the current task is complete."""
    message: str = Field(default="Task completed", description="Completion message")


class AsyncToolExecutor:
    """Executes tools for an LLM agent asynchronously."""

    def __init__(self, game_client: AsyncGameClient, character_id: str):
        """Initialize the async tool executor.
        
        Args:
            game_client: Async client for game server API calls
            character_id: ID of the character being controlled
        """
        self.game_client = game_client
        self.character_id = character_id
        self.finished = False
        self.finished_message = ""

    def _build_sector_info(self, contents: Optional[Any]) -> Dict[str, Any]:
        """Create a standardized sector info dictionary.

        Args:
            contents: Optional sector contents from the server.

        Returns:
            Dictionary with port information, other players, and adjacent sectors.
        """
        sector_info: Dict[str, Any] = {}
        if contents:
            port_info = None
            if getattr(contents, "port", None):
                port = contents.port
                port_info = {
                    "class": port.class_num,
                    "code": port.code,
                    "buys": port.buys,
                    "sells": port.sells,
                    "stock": port.stock,
                    "demand": port.demand,
                }
            sector_info["port_info"] = port_info
            players = getattr(contents, "other_players", []) or []
            try:
                # Handle both cases: list of player objects or list of strings
                if players and hasattr(players[0], 'name'):
                    sector_info["other_players"] = [player.name for player in players]
                else:
                    # Already a list of strings (player names)
                    sector_info["other_players"] = players if isinstance(players, list) else []
            except (TypeError, AttributeError, IndexError):
                sector_info["other_players"] = []
            adjacent = getattr(contents, "adjacent_sectors", [])
            if not isinstance(adjacent, list):
                adjacent = []
            sector_info["adjacent_sectors"] = adjacent
        return sector_info

    async def plot_course(self, from_sector: int, to_sector: int) -> Dict[str, Any]:
        """Execute plot course tool."""
        try:
            result = await self.game_client.plot_course(from_sector, to_sector)
            return {
                "success": True,
                "path": result.path,
                "distance": result.distance,
                "from_sector": result.from_sector,
                "to_sector": result.to_sector
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def move(self, to_sector: int) -> Dict[str, Any]:
        """Execute move tool."""
        try:
            # Get the old sector from the client's tracked position
            old_sector = self.game_client.current_sector
            
            # Validate that we know our current position
            if old_sector is None:
                return {
                    "success": False,
                    "error": "Current sector unknown. Run my_status first to establish position."
                }

            result = await self.game_client.move(self.character_id, to_sector)

            # Build sector contents info
            sector_info = self._build_sector_info(result.sector_contents)
            
            return {
                "success": True,
                "old_sector": old_sector,
                "new_sector": result.sector,
                "character_id": result.id,
                "sector_contents": sector_info
            }
        except Exception as e:
            # Try to extract more detailed error information
            error_msg = str(e)
            if hasattr(e, '__cause__') and e.__cause__:
                error_msg = f"{error_msg} (Caused by: {str(e.__cause__)})"
            
            return {
                "success": False,
                "error": error_msg
            }
    
    async def my_status(self) -> Dict[str, Any]:
        """Execute my-status tool."""
        try:
            result = await self.game_client.my_status()

            # Build sector contents info
            sector_info = self._build_sector_info(result.sector_contents)
            
            return {
                "success": True,
                "current_sector": result.sector,
                "character_id": result.id,
                "sector_contents": sector_info
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def wait_for_time(self, seconds: float) -> Dict[str, Any]:
        """Execute wait tool."""
        await asyncio.sleep(seconds)
        return {
            "success": True,
            "waited_seconds": seconds
        }
    
    async def my_map(self) -> Dict[str, Any]:
        """Execute my-map tool."""
        try:
            response = await self.game_client.my_map()
            return {
                "success": True,
                **response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def find_port(self, commodity: str = None, buy_or_sell: str = None, from_sector: int = None) -> Dict[str, Any]:
        """Execute find-port tool."""
        try:
            # Use current sector if from_sector not specified
            if from_sector is None:
                from_sector = self.game_client.current_sector
                if from_sector is None:
                    return {
                        "success": False,
                        "error": "No current sector tracked. Run my_status first."
                    }
            
            # If no commodity specified, find ANY nearest port
            if commodity is None:
                result = await self.game_client.find_nearest_known_port(from_sector)
                
                if result is None:
                    return {
                        "success": True,
                        "found": False,
                        "message": f"No known ports found from sector {from_sector}"
                    }
                
                return {
                    "success": True,
                    "found": True,
                    **result
                }
            
            # If commodity specified, buy_or_sell is required
            if buy_or_sell is None:
                return {
                    "success": False,
                    "error": "buy_or_sell is required when commodity is specified"
                }
            
            # Find port with specific commodity
            result = await self.game_client.find_nearest_known_port_with_commodity(
                from_sector,
                commodity,
                buy_or_sell
            )
            
            if result is None:
                return {
                    "success": True,
                    "found": False,
                    "message": f"No known port that {buy_or_sell}s {commodity}"
                }
            
            return {
                "success": True,
                "found": True,
                **result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def finish_task(self, message: str = "Task completed") -> Dict[str, Any]:
        """Execute finished tool."""
        self.finished = True
        self.finished_message = message
        return {
            "success": True,
            "message": message
        }
    
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        tool_map = {
            "plot_course": self.plot_course,
            "move": self.move,
            "my_status": self.my_status,
            "my_map": self.my_map,
            "find_port": self.find_port,
            "wait_for_time": self.wait_for_time,
            "finished": self.finish_task,
        }
        
        if tool_name not in tool_map:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
        
        try:
            # Execute the async tool function with provided arguments
            return await tool_map[tool_name](**tool_args)
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get OpenAI function calling format tool definitions.
    
    Returns:
        List of tool definitions for OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "plot_course",
                "description": "Calculate the shortest path between two sectors in the game universe",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_sector": {
                            "type": "integer",
                            "description": "Starting sector ID"
                        },
                        "to_sector": {
                            "type": "integer",
                            "description": "Destination sector ID"
                        }
                    },
                    "required": ["from_sector", "to_sector"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move your ship to an adjacent sector. You can only move one sector at a time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_sector": {
                            "type": "integer",
                            "description": "Adjacent sector ID to move to"
                        }
                    },
                    "required": ["to_sector"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "my_status",
                "description": "Get your current status including current sector position",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "my_map",
                "description": "Get your map knowledge including all visited sectors, known ports, and discovered connections",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_port",
                "description": "Find the nearest known port. Can optionally filter by commodity type. If no parameters given, finds ANY nearest port.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_sector": {
                            "type": "integer",
                            "description": "Optional: Sector to search from (defaults to current sector)"
                        },
                        "commodity": {
                            "type": "string",
                            "description": "Optional: The commodity to search for",
                            "enum": ["fuel_ore", "organics", "equipment"]
                        },
                        "buy_or_sell": {
                            "type": "string",
                            "description": "Optional: Whether to find a port that 'buy's or 'sell's the commodity (required if commodity is specified)",
                            "enum": ["buy", "sell"]
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wait_for_time",
                "description": "Wait for a specified number of seconds before continuing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "number",
                            "description": "Number of seconds to wait (max 60)",
                            "minimum": 0,
                            "maximum": 60
                        }
                    },
                    "required": ["seconds"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finished",
                "description": "Signal that you have completed the assigned task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Completion message describing what was accomplished",
                            "default": "Task completed"
                        }
                    }
                }
            }
        }
    ]
