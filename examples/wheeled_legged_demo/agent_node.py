"""
AgentNode for wheeled-legged robot.
Uses Remote Ollama Gemma4:e4b with tool calling for memory-based reasoning.
Publishes navigation goal poses to /goal_pose.
"""

import json
import math
import traceback
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String

from ollama_client import OllamaClient
from lite_memory import LiteMemory


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_text",
            "description": (
                "Search robot's visual memory by text description. "
                "Use this to find places, objects, or scenes the robot has seen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A descriptive phrase to search for, e.g. "
                            "'a desk with snacks' or 'a red sofa'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_position",
            "description": (
                "Search robot's memory by spatial position (x, y, z)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate"},
                    "y": {"type": "number", "description": "Y coordinate"},
                    "z": {"type": "number", "description": "Z coordinate (usually 0)"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_time",
            "description": "Search robot's memory by time in HH:MM:SS format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM:SS format, e.g. '14:30:00'",
                    }
                },
                "required": ["time"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful robot assistant with access to a spatio-temporal memory database.
When a user asks about places, objects, or events, use the retrieval tools to search your memory.

After gathering information, respond with a JSON object (no markdown fencing) containing:
{
  "text": "Your natural language answer",
  "position": [x, y, z] or null,
  "orientation": angle_in_radians or null,
  "time": "HH:MM:SS" or null,
  "binary": "yes" or "no" or null,
  "duration": seconds or null
}

If the user asks to go somewhere or be taken to a location, set "position" to the target coordinates.
If you cannot determine a value, set it to null."""


class AgentNode(Node):

    def __init__(self):
        super().__init__("AgentNode")

        self.declare_parameter("ollama_host", "192.168.31.63")
        self.declare_parameter("ollama_port", 11434)
        self.declare_parameter("llm_model", "gemma4:e4b")
        self.declare_parameter("db_path", "./remembr_memory.db")
        self.declare_parameter("db_collection", "robot_memory")
        self.declare_parameter("query_topic", "/speech")
        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("query_keyword", "robot")

        ollama_host = self.get_parameter("ollama_host").value
        ollama_port = self.get_parameter("ollama_port").value

        self.ollama = OllamaClient(host=ollama_host, port=ollama_port)
        self.llm_model = self.get_parameter("llm_model").value

        self.memory = LiteMemory(
            db_path=self.get_parameter("db_path").value,
            collection_name=self.get_parameter("db_collection").value,
            ollama_host=ollama_host,
            ollama_port=ollama_port,
        )

        keyword = self.get_parameter("query_keyword").value
        if keyword:
            self.query_filter = lambda text: keyword.lower() in text.lower()
        else:
            self.query_filter = lambda text: True

        self.query_subscriber = self.create_subscription(
            String,
            self.get_parameter("query_topic").value,
            self.query_callback,
            10,
        )
        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10,
        )
        self.goal_pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_parameter("goal_pose_topic").value,
            10,
        )

        self.last_pose = None
        self.logger = self.get_logger()
        self.logger.info(f"AgentNode ready. LLM: {self.llm_model}")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.last_pose = msg

    def query_callback(self, msg: String):
        if not self.query_filter(msg.data):
            self.logger.info(f"Skipping query (no keyword): {msg.data[:60]}")
            return

        self.logger.info(f"Processing query: {msg.data}")

        try:
            result = self._run_agent(msg.data)
            self.logger.info(f"Agent result: {result}")

            position = result.get("position")
            if position and len(position) >= 2:
                self._publish_goal_pose(position, result.get("orientation"))

            text_response = result.get("text", "")
            if text_response:
                self.logger.info(f"Answer: {text_response}")

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            self.logger.error(traceback.format_exc())

    def _run_agent(self, query: str, max_tool_rounds: int = 3) -> dict:
        context = ""
        if self.last_pose:
            from memory_builder_node import pose_msg_to_values
            pos, angle, t = pose_msg_to_values(self.last_pose)
            context = f"\nYou are currently at position {np.array(pos).round(2).tolist()}, orientation {angle:.2f} rad."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query + context},
        ]

        for round_i in range(max_tool_rounds):
            response = self.ollama.chat_with_tools(
                messages=messages,
                tools=TOOLS,
                model=self.llm_model,
            )

            tool_calls = response.get("tool_calls")
            if not tool_calls:
                return self._parse_response(response.get("content", ""))

            messages.append({"role": "assistant", "content": response.get("content", ""), "tool_calls": tool_calls})

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                self.logger.info(f"  Tool call [{round_i}]: {fn_name}({fn_args})")

                tool_result = self._execute_tool(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                })

        final = self.ollama.chat(messages=messages, model=self.llm_model)
        return self._parse_response(final)

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "retrieve_from_text":
                return self.memory.search_by_text(args["query"])
            elif name == "retrieve_from_position":
                pos = (args.get("x", 0), args.get("y", 0), args.get("z", 0))
                return self.memory.search_by_position(pos)
            elif name == "retrieve_from_time":
                return self.memory.search_by_time(args["time"])
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool error: {e}"

    def _parse_response(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        import re
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {
            "text": text,
            "position": None,
            "orientation": None,
            "time": None,
            "binary": None,
            "duration": None,
        }

    def _publish_goal_pose(self, position: list, orientation: float = None):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(position[0])
        goal.pose.position.y = float(position[1])
        goal.pose.position.z = float(position[2]) if len(position) > 2 else 0.0

        if orientation is not None:
            yaw = float(orientation)
        else:
            yaw = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pose_publisher.publish(goal)
        self.logger.info(
            f"Published goal_pose: ({position[0]:.2f}, {position[1]:.2f}), yaw={yaw:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
