from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat import GroupChat, GroupChatManager
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


llm_config = {
    "config_list": [{
        "model": "gemini-1.5-flash",
        "api_key": api_key,
        "api_type": "google"
    }],
    "timeout": 120
}

user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
    llm_config = False,
    system_message="A human admin that provides the story topic and reviews final output.",
)


planning_agent = AssistantAgent(
    name="Planning_Agent",
    system_message="""As Story Workflow Manager, oversee this process:

        Confirm topic with User_Proxy
        Direct Story_Writer's draft
        Engage Story_Reviewer for feedback
        Manage revisions
        Initiate Moral_Extractor when ready
        Deliver final story
        Enforce workflow order and completion.""",
    llm_config=llm_config,
    
)


story_writer = AssistantAgent(
    name="Story_Writer",
    system_message="""
    You are a creative Story Writer. You only write when directed by Planning_Agent.
    Create a story on the topic provided (400-600 words) and revise it on the feedback from Story_Reviewer
    
    Your stories must include:
    - Clear narrative structure
    - Meaningful
    - Appropriate pacing
    
    Format stories with clear paragraphs.
    Wait for explicit instructions from Planning_Agent before writing.
    """,
    llm_config=llm_config,
    
    
)


story_reviewer = AssistantAgent(
    name="Story_Reviewer",
    system_message="""
    You are a Story Reviewer. You ONLY provide feedback when directed by Planning_Agent.
    
    When instructed:
    1. Analyze the story thoroughly
    2. Provide structured feedback:
       - Overall impression
       - Strengths (3-5 points)
       - Areas for improvement (3-5 points)
       - Specific suggestions
    3. Focus on:
       - Plot coherence
       - Character development
       - Pacing
       - Theme development
    You should be constructive and professional.
    Wait for Planning_Agent's instruction before reviewing.
    """,
    llm_config=llm_config,
    
)


moral_extractor = AssistantAgent(
    name="Moral_Extractor",
    system_message="""
    You extract morals from completed stories WHEN DIRECTED by Planning_Agent.

    When instructed:
    1. Analyze the final story version
    2. Identify core moral/lesson (1-2 sentences)
    3. Explain how story conveys this moral
    4. Format as:
       Moral: [clear statement]
       Explanation: [brief analysis]
    
    Only respond when explicitly asked by Planning_Agent.
    """,
    llm_config=llm_config,
    
)

planning_agent.memory = ListMemory()
story_writer.memory = ListMemory()
story_reviewer.memory = ListMemory()
moral_extractor.memory = ListMemory()

import asyncio
async def preload_memory():
    await story_writer.memory.add(
        MemoryContent(content="Always write vivid and imaginative stories.", mime_type=MemoryMimeType.TEXT)
    )
    await story_reviewer.memory.add(
        MemoryContent(content="Focus on constructive critique with examples.", mime_type=MemoryMimeType.TEXT)
    )
    await moral_extractor.memory.add(
        MemoryContent(content="Try to link morals to real-world values.", mime_type=MemoryMimeType.TEXT)
    )


agents = [user_proxy, planning_agent, story_writer, story_reviewer, moral_extractor]

groupchat = GroupChat(
    agents=agents,
    messages=[],
    max_round=30,
    speaker_selection_method="round_robin",
    allow_repeat_speaker=True,
)


manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    system_message="""
    You facilitate the story creation workflow. Support Planning_Agent by:
    1. Ensuring proper speaker selection
    2. Maintaining conversation focus
    3. Preventing off-topic discussions
    4. Enforcing the workflow stages
    5. Terminating when Planning_Agent declares completion
    
    Prioritize Planning_Agent's instructions.
    """,
)

def initiate_story_creation(topic):
    """Start the managed story creation process"""
    
    for agent in agents:
        agent.reset()
    
    user_proxy.initiate_chat(
        manager,
        message=f"Let's create a story about: {topic}\n\nPlanning_Agent, please manage the workflow.",
    )

if __name__ == "__main__":
    print("=== Story Creation System (Google Gemini) ===")
    print("Enter a topic for your story (e.g., 'perseverance', 'friendship', etc.)")
    topic = input("Story topic: ")

    asyncio.run(preload_memory())
    
    print(f"\nCreating story about: {topic}...\n")
    initiate_story_creation(topic)
    
    print("\n=== Process Complete ===")