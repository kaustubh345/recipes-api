from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
import dotenv
import os
import asyncio
from typing import Any
from github import Github
from github import Auth
from github import GithubException
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context


dotenv.load_dotenv()   

print(os.getenv("OPENAI_MODEL"))
repo_url = os.getenv("REPO_URL")

pr_number = int(os.getenv("PR_NUMBER"))

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

current_state= {}

async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "Saved gathered context to state."

async  def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Saved draft comment to state."

async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "Saved final review to state."

def post_review_to_github(pr_number: int, comment: str) -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set in environment variables.")
    repo_full_name = os.getenv("REPO_FULL_NAME")
    if not repo_full_name:
        raise RuntimeError("REPO_FULL_NAME is not set in environment variables.")
    g = Github(auth=Auth.Token(token))
    repo = g.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)
    pr.create_review(body=comment, event="COMMENT")
    return f"Posted review to PR #{pr_number} in {repo_full_name}."

post_review_to_github_tool = FunctionTool.from_defaults(
    post_review_to_github,
)


review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the drafted PR comment, requests rewrites if needed, stores final review, and posts it to GitHub.",
    system_prompt="""You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
    
    Once a review is generated, you need to run a final check and post it to GitHub.
    
    The review must:
    Be a ~200-300 word review in markdown format.
    Specify what is good about the PR:
    Did the author follow ALL contribution rules? What is missing?
    Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
    Are there notes on whether new endpoints were documented?
    Are there suggestions on which lines could be improved upon? Are these lines quoted?
    If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
    When you are satisfied, post the review to GitHub.""",
    tools=[add_final_review_to_state, post_review_to_github_tool],
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        system_prompt="""You are the commentor agent that writes review comments for pull requests as a human reviewer would. 
    Ensure to do the following for a thorough review: 
    - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
    - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
        - What is good about the PR?
        - Did the author follow ALL contribution rules? What is missing?
        - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
        - Are new endpoints documented? - use the diff to determine this.
        - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
        - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
        Keep existing instructions about requesting context from ContextAgent.
    - If you need any additional details, you must hand off to the ContextAgent.
    - After drafting the review, you MUST call add_comment_to_state with the complete draft.
    - After saving the draft, you MUST call handoff to ReviewAndPostingAgent.
    - Do NOT end with a final response yourself. ReviewAndPostingAgent is responsible for final review and posting.
    - You should directly address the author. So your comments should sound like:
    "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?""",
        tools=[add_comment_to_state],
        can_handoff_to=["ContextAgent","ReviewAndPostingAgent"],
)


def get_repo_details(repo_full_name: str):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set in environment variables.")

    # For github.com, PyGithub should use the default API endpoint.
    g = Github(auth=Auth.Token(token))
    repo = g.get_repo(repo_full_name)

    return {
        "full_name": repo.full_name,
        "description": repo.description,
        "private": repo.private,
        "default_branch": repo.default_branch,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "open_issues": repo.open_issues_count,
        "url": repo.html_url,
        "Open PRs": repo.get_pulls(state="open").totalCount
    }

def get_pr_details(pr_number: int):
    """ get details of a pull request by its number """
    print(f"Fetching details for PR #{pr_number}")
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set in environment variables.")

    g = Github(auth=Auth.Token(token))
    repo_full_name = os.getenv("REPO_FULL_NAME")
    if not repo_full_name:
        raise RuntimeError("REPO_FULL_NAME is not set in environment variables.")
    print(f"Fetching details for repository '{repo_full_name}'")
    repo = g.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)

    print(f"Fetched details for PR #{pr_number}")

    commit_SHAs = []
    commits = pr.get_commits()

    for c in commits:
        commit_SHAs.append(c.sha)

    return {
        "author": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "commit_sha": pr.head.sha,
        "all_commit_shas": commit_SHAs,
    }


def files_details(git_file_path: str):
    """
    given a file path, this tool can fetch the contents of the file from the git repository.
    :param git_file_path:
    :return:
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set in environment variables.")

    g = Github(auth=Auth.Token(token))
    repo_full_name = os.getenv("REPO_FULL_NAME")
    if not repo_full_name:
        raise RuntimeError("REPO_FULL_NAME is not set in environment variables.")

    repo = g.get_repo(repo_full_name)
    try:
        file_content = repo.get_contents(git_file_path)
        return file_content.decoded_content.decode("utf-8")
    except GithubException as exc:
        print(f"GitHub API error: {exc.status} {exc.data}")
        return None


def pr_commits_details(commit_sha: str):
    """
    given the commit SHA, this function can retrieve information about the commit, such as the files that changed, and return that information.

    :param commit_sha: The SHA of the commit to fetch details for.
    :return: A dictionary containing details about the commit, or None if an error occurs.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set in environment variables.")

    g = Github(auth=Auth.Token(token))
    repo_full_name = os.getenv("REPO_FULL_NAME")
    if not repo_full_name:
        raise RuntimeError("REPO_FULL_NAME is not set in environment variables.")

    repo = g.get_repo(repo_full_name)
    try:
        commit = repo.get_commit(commit_sha)
        changed_files: list[dict[str, Any]] = []
        for f in commit.files:
            changed_files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            })
        return changed_files
    except GithubException as exc:
        print(f"GitHub API error: {exc.status} {exc.data}")
        return None

get_pr_details_tool = FunctionTool.from_defaults(
    get_pr_details,
)

pr_commits_details_tool = FunctionTool.from_defaults(
    pr_commits_details,
)

files_details_tool = FunctionTool.from_defaults(
    files_details,
)





"""
create the context manager agent that will orchestrate the whole context 
retrieval process. You need to equip this agent with the tools 
(not the functions) you created and the LLM to use. 
You should use the ReAct agent class to create the agent. You should use the following system prompt for this agent:
"""

#contextManagerAgent creation
ContextAgent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all needed pull request context including PR details, changed files, and requested repository files.",
    system_prompt="You are the context gathering agent. When gathering context, you MUST gather \n: \
        The details: author, title, body, diff_url, state, and head_sha; \n\
        Changed files; \n\
        Any requested for files; \n\
        Once you gather the requested info, you MUST hand control back to the Commentor Agent.",
    tools=[get_pr_details_tool, pr_commits_details_tool, files_details_tool, add_context_to_state],
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[ContextAgent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
    "gathered_contexts": "",
    "draft_comment": "",
    "final_review": "",
    },
)

async def main():
    try:
        query = "Write a review for PR: " + str(pr_number)
    except EOFError:
        print("No input provided. Example: Review PR #1, then finalize and post the review.")
        return

    if not query:
        print("No input provided. Example: Review PR #1, then finalize and post the review.")
        return

    query += (
        "\n\nWorkflow requirements (must follow): "
        "ContextAgent gathers context and saves it with add_context_to_state. "
        "CommentorAgent drafts the review, must call add_comment_to_state, then must handoff to ReviewAndPostingAgent. "
        "ReviewAndPostingAgent performs final check, must call add_final_review_to_state, and then call post_review_to_github."
    )

    prompt = RichPromptTemplate(query)

    async def stream_handler(handler):
        current_agent = None
        last_agent = None
        last_response = None

        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"Current agent: {current_agent}")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    last_agent = current_agent
                    last_response = event.response.content
                    print("\n\nFinal response:", event.response.content)
                if event.tool_calls:
                    print("Selected tools: ", [call.tool_name for call in event.tool_calls])
            elif isinstance(event, ToolCallResult):
                print(f"Output from tool: {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

        return last_agent, last_response

    handler = workflow_agent.run(prompt.format())
    last_agent, last_response = await stream_handler(handler)

    # Fallback: some models stop at CommentorAgent without performing the required handoff.
    if last_agent == "CommentorAgent" and last_response:
        print("\nFallback: continuing workflow in ReviewAndPostingAgent to finalize and post the review.")
        follow_up = (
            "Continue from the draft review below. You MUST perform final review, "
            "call add_final_review_to_state, and call post_review_to_github.\n\n"
            f"Draft review:\n{last_response}"
        )
        follow_up_handler = workflow_agent.run(follow_up)
        await stream_handler(follow_up_handler)
if __name__ == "__main__":
    asyncio.run(main())