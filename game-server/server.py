#!/usr/bin/env python3
"""
Gradient Bang game server - FastAPI backend for a TradeWars-inspired space game.
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import deque

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from character_knowledge import CharacterKnowledgeManager


class UniverseGraph:
    """Graph representation of the game universe for efficient pathfinding."""

    def __init__(self, universe_data: dict):
        self.sector_count = universe_data["meta"]["sector_count"]
        self.adjacency: Dict[int, List[int]] = {}

        # Build adjacency list from universe structure
        for sector in universe_data["sectors"]:
            sector_id = sector["id"]
            self.adjacency[sector_id] = []

            for warp in sector["warps"]:
                self.adjacency[sector_id].append(warp["to"])

    def find_path(self, start: int, end: int) -> Optional[List[int]]:
        """Find shortest path using BFS."""
        if start == end:
            return [start]

        if start not in self.adjacency or end not in self.adjacency:
            return None

        visited: Set[int] = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            for neighbor in self.adjacency.get(current, []):
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


# Pydantic Models for API


class PortInfo(BaseModel):
    """Port information for client display."""

    class_num: int = Field(..., alias="class")
    code: str
    buys: List[str]
    sells: List[str]
    stock: Dict[str, int]
    stock_max: Dict[str, int]
    demand: Dict[str, int]
    demand_max: Dict[str, int]


class PlanetInfo(BaseModel):
    """Planet information for client display."""

    id: str
    class_code: str
    class_name: str


class SectorContents(BaseModel):
    """Contents of a sector visible to players."""

    port: Optional[PortInfo] = None
    planets: List[PlanetInfo] = []
    other_players: List[str] = []  # List of players in this sector
    adjacent_sectors: List[int] = []  # List of sectors you can warp to


class Character:
    """Represents a player character in the game."""

    def __init__(self, character_id: str, sector: int = 0):
        self.id = character_id
        self.sector = sector
        self.last_active = datetime.now(timezone.utc)

    def update_activity(self):
        """Update the last active timestamp."""
        self.last_active = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert character to dictionary for API responses."""
        return {"id": self.id, "sector": self.sector, "last_active": self.last_active.isoformat()}


class ConnectionManager:
    """Manages WebSocket connections for the firehose."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.broadcast_task = None

    async def connect(self, websocket: WebSocket):
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "message": "Connected to Gradient Bang firehose",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_event(self, event: dict):
        """Add an event to the queue for broadcasting."""
        await self.event_queue.put(event)

    async def _broadcast_worker(self):
        """Background task to broadcast events from the queue."""
        while True:
            try:
                event = await self.event_queue.get()
                # Send to all connected clients
                disconnected = []
                for connection in self.active_connections:
                    try:
                        await connection.send_json(event)
                    except:
                        disconnected.append(connection)

                # Remove disconnected clients
                for conn in disconnected:
                    self.disconnect(conn)

            except Exception as e:
                print(f"Broadcast error: {e}")
                await asyncio.sleep(0.1)

    def start_broadcast_task(self):
        """Start the background broadcast task."""
        if self.broadcast_task is None:
            self.broadcast_task = asyncio.create_task(self._broadcast_worker())


class GameWorld:
    """Container for all game world data."""

    def __init__(self):
        self.universe_graph: Optional[UniverseGraph] = None
        self.sector_contents: Optional[dict] = None
        self.characters: Dict[str, Character] = {}
        self.connection_manager = ConnectionManager()
        self.knowledge_manager = CharacterKnowledgeManager()

    def get_sector_contents(
        self, sector_id: int, current_character_id: str = None
    ) -> SectorContents:
        """Get the contents of a sector visible to a player.

        Args:
            sector_id: The sector to examine
            current_character_id: The character making the query (to exclude from other_players)

        Returns:
            SectorContents object with all visible information
        """
        # Get port information if present
        port_info = None
        if self.sector_contents and sector_id < len(self.sector_contents["sectors"]):
            sector_data = self.sector_contents["sectors"][sector_id]
            if sector_data.get("port"):
                port_data = sector_data["port"]
                port_info = PortInfo(
                    **port_data  # Pass the entire port data dict, Pydantic will handle the alias
                )

        # Get planets if present
        planets = []
        if self.sector_contents and sector_id < len(self.sector_contents["sectors"]):
            sector_data = self.sector_contents["sectors"][sector_id]
            for planet_data in sector_data.get("planets", []):
                planets.append(
                    PlanetInfo(
                        id=planet_data["id"],
                        class_code=planet_data["class_code"],
                        class_name=planet_data["class_name"],
                    )
                )

        # Get other players in this sector
        other_players = []
        for char_id, character in self.characters.items():
            if character.sector == sector_id and char_id != current_character_id:
                other_players.append(char_id)

        # Get adjacent sectors
        adjacent_sectors = []
        if self.universe_graph and sector_id in self.universe_graph.adjacency:
            adjacent_sectors = sorted(self.universe_graph.adjacency[sector_id])

        return SectorContents(
            port=port_info,
            planets=planets,
            other_players=other_players,
            adjacent_sectors=adjacent_sectors,
        )

    def load_data(self):
        """Load universe data from JSON files."""
        # Find world-data directory relative to server file
        server_dir = Path(__file__).parent
        world_data_path = server_dir.parent / "world-data"

        # Load universe structure
        universe_path = world_data_path / "universe_structure.json"
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe structure file not found: {universe_path}")

        with open(universe_path, "r") as f:
            universe_data = json.load(f)

        self.universe_graph = UniverseGraph(universe_data)

        # Load sector contents
        contents_path = world_data_path / "sector_contents.json"
        if not contents_path.exists():
            raise FileNotFoundError(f"Sector contents file not found: {contents_path}")

        with open(contents_path, "r") as f:
            self.sector_contents = json.load(f)

        print(f"Loaded universe with {self.universe_graph.sector_count} sectors")


# Initialize game world
game_world = GameWorld()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    try:
        game_world.load_data()
        game_world.connection_manager.start_broadcast_task()
        print("Game world loaded successfully")
        print("Firehose broadcast task started")
    except Exception as e:
        print(f"Failed to load game world: {e}")
        raise

    yield

    # Shutdown (nothing to do yet)
    pass


app = FastAPI(title="Gradient Bang", version="0.1.0", lifespan=lifespan)


class PlotCourseRequest(BaseModel):
    """Request model for plot-course endpoint."""

    from_sector: int = Field(..., alias="from", ge=0)
    to_sector: int = Field(..., alias="to", ge=0)


class PlotCourseResponse(BaseModel):
    """Response model for plot-course endpoint."""

    from_sector: int
    to_sector: int
    path: List[int]
    distance: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Gradient Bang",
        "version": "0.1.0",
        "status": "running",
        "sectors": game_world.universe_graph.sector_count if game_world.universe_graph else 0,
    }


@app.post("/api/plot-course", response_model=PlotCourseResponse)
async def plot_course(request: PlotCourseRequest):
    """Calculate shortest path between two sectors."""
    if not game_world.universe_graph:
        raise HTTPException(status_code=503, detail="Game world not loaded")

    # Validate sector IDs
    if request.from_sector >= game_world.universe_graph.sector_count:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid from_sector: {request.from_sector}. Must be < {game_world.universe_graph.sector_count}",
        )

    if request.to_sector >= game_world.universe_graph.sector_count:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid to_sector: {request.to_sector}. Must be < {game_world.universe_graph.sector_count}",
        )

    # Find path
    path = game_world.universe_graph.find_path(request.from_sector, request.to_sector)

    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No path found from sector {request.from_sector} to sector {request.to_sector}",
        )

    return PlotCourseResponse(
        from_sector=request.from_sector,
        to_sector=request.to_sector,
        path=path,
        distance=len(path) - 1,
    )


class JoinRequest(BaseModel):
    """Request model for join endpoint."""

    character_id: str = Field(..., min_length=1, max_length=100)


class CharacterStatus(BaseModel):
    """Response model for character status with sector contents."""

    id: str
    sector: int
    last_active: str
    sector_contents: SectorContents


@app.post("/api/join", response_model=CharacterStatus)
async def join(request: JoinRequest):
    """Add a new character to the game at sector 0."""
    is_new = request.character_id not in game_world.characters

    if is_new:
        # Create new character at sector 0
        character = Character(request.character_id, sector=0)
        game_world.characters[request.character_id] = character

        # Broadcast join event
        join_event = {
            "type": "join",
            "character_id": request.character_id,
            "sector": 0,
            "timestamp": character.last_active.isoformat(),
        }
        asyncio.create_task(game_world.connection_manager.broadcast_event(join_event))
    else:
        # Character already exists, return their current status
        character = game_world.characters[request.character_id]
        character.update_activity()

    # Get sector contents for the character's current position
    sector_contents = game_world.get_sector_contents(character.sector, request.character_id)

    # Update character's map knowledge
    port_info = None
    if sector_contents.port:
        port_info = sector_contents.port.model_dump()

    planets = [planet.model_dump() for planet in sector_contents.planets]

    game_world.knowledge_manager.update_sector_visit(
        character_id=request.character_id,
        sector_id=character.sector,
        port_info=port_info,
        planets=planets,
        adjacent_sectors=sector_contents.adjacent_sectors,
    )

    return CharacterStatus(**character.to_dict(), sector_contents=sector_contents)


class MoveRequest(BaseModel):
    """Request model for move endpoint."""

    character_id: str = Field(..., min_length=1, max_length=100)
    to: int = Field(..., ge=0)


@app.post("/api/move", response_model=CharacterStatus)
async def move(request: MoveRequest):
    """Move a character to an adjacent sector."""
    if not game_world.universe_graph:
        raise HTTPException(status_code=503, detail="Game world not loaded")

    # Check if character exists
    if request.character_id not in game_world.characters:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{request.character_id}' not found. Join the game first.",
        )

    character = game_world.characters[request.character_id]
    current_sector = character.sector

    # Validate destination sector
    if request.to >= game_world.universe_graph.sector_count:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sector: {request.to}. Must be < {game_world.universe_graph.sector_count}",
        )

    # Check if destination is adjacent
    adjacent_sectors = game_world.universe_graph.adjacency.get(current_sector, [])
    if request.to not in adjacent_sectors:
        raise HTTPException(
            status_code=400,
            detail=f"Sector {request.to} is not adjacent to current sector {current_sector}",
        )

    # Move the character
    old_sector = character.sector
    character.sector = request.to
    character.update_activity()

    # Broadcast movement event to firehose
    movement_event = {
        "type": "movement",
        "character_id": request.character_id,
        "from_sector": old_sector,
        "to_sector": request.to,
        "timestamp": character.last_active.isoformat(),
    }
    # Use create_task to avoid blocking the response
    asyncio.create_task(game_world.connection_manager.broadcast_event(movement_event))

    # Get sector contents for the new position
    sector_contents = game_world.get_sector_contents(character.sector, request.character_id)

    # Update character's map knowledge for the new sector
    port_info = None
    if sector_contents.port:
        port_info = sector_contents.port.model_dump()

    planets = [planet.model_dump() for planet in sector_contents.planets]

    game_world.knowledge_manager.update_sector_visit(
        character_id=request.character_id,
        sector_id=character.sector,
        port_info=port_info,
        planets=planets,
        adjacent_sectors=sector_contents.adjacent_sectors,
    )

    return CharacterStatus(**character.to_dict(), sector_contents=sector_contents)


class StatusRequest(BaseModel):
    """Request model for my-status endpoint."""

    character_id: str = Field(..., min_length=1, max_length=100)


@app.post("/api/my-status", response_model=CharacterStatus)
async def my_status(request: StatusRequest):
    """Get the current status of a character."""
    if request.character_id not in game_world.characters:
        raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")

    character = game_world.characters[request.character_id]
    character.update_activity()

    # Get sector contents for the character's current position
    sector_contents = game_world.get_sector_contents(character.sector, request.character_id)

    return CharacterStatus(**character.to_dict(), sector_contents=sector_contents)


class MapRequest(BaseModel):
    """Request model for my-map endpoint."""

    character_id: str = Field(..., min_length=1, max_length=100)


@app.post("/api/my-map")
async def my_map(request: MapRequest):
    """Get the map knowledge for a character."""
    knowledge = game_world.knowledge_manager.load_knowledge(request.character_id)
    return knowledge.model_dump()


@app.websocket("/api/firehose")
async def websocket_firehose(websocket: WebSocket):
    """WebSocket endpoint for real-time game events."""
    await game_world.connection_manager.connect(websocket)
    try:
        # Keep the connection alive
        while True:
            # Wait for any message from client (like ping/pong)
            await websocket.receive_text()
    except WebSocketDisconnect:
        game_world.connection_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
