from importlib import resources
from marstrace.error import MarsTraceError


class AgentGuide:
    """The skills that teach an agent to use MarsTrace, as the package ships them.

    An agent that has the Claude Code plugin loads them as skills; any other agent reads them
    through ``marstrace guide`` or the MCP tool ``marstrace_guide``.
    """

    # topic -> the skill that covers it
    MAP_TOPIC_SKILL = {
        "remote": "marstrace-remote",
        "read": "marstrace-read",
    }
    MAP_TOPIC_SUMMARY = {
        "remote": "set up an account and a key, gather a workspace, estimate, submit and fetch a job",
        "read": "gate a fetched trace and read it with marstrace show",
    }
    SKILL_FILE = "SKILL.md"
    FRONTMATTER_FENCE = "---"

    CONSOLE_URL = "https://console.marscompute.ai"
    # the steps that give a user a MarsTrace key with credit behind it, as the console orders them
    SETUP_STEPS = (
        "MarsTrace needs a MarsCompute account with credit and a MarsTrace API key. Walk the user "
        "through it; never ask them to paste the key into the chat:\n"
        f"1. Register at {CONSOLE_URL}/register (an email code or Google; no approval).\n"
        f"2. Save the billing profile at {CONSOLE_URL}/tenant/billing/profile.\n"
        f"3. Add credit at {CONSOLE_URL}/tenant/billing/credits; a job starts only while the "
        "balance reaches the minimum that marstrace credit shows.\n"
        f"4. Create a key at {CONSOLE_URL}/tenant/trace/keys. It starts with mt_live_; the "
        "console's other API keys do not work here.\n"
        "5. The user sets it in the shell that starts the agent, e.g. "
        "export MARSTRACE_API_KEY=mt_live_..., then restarts the agent so the MCP server reads it.\n"
        "Read the whole workflow with marstrace guide remote (MCP: marstrace_guide, topic remote)."
    )

    @classmethod
    def topics(cls) -> list[str]:
        """The topics a guide can be read on.

        Returns:
            the topic names, in the order an agent needs them
        """
        return list(cls.MAP_TOPIC_SKILL)

    @classmethod
    def text(cls, topic: str) -> str:
        """The guide on one topic: its skill without the skill's front matter.

        Args:
            topic: a name from ``topics()``

        Returns:
            the guide, as Markdown

        Raises:
            MarsTraceError: the topic is not one the package has a guide on
        """
        skill = cls.MAP_TOPIC_SKILL.get(topic)
        if skill is None:
            raise MarsTraceError(
                f"no guide on {topic!r}; the topics are {', '.join(cls.topics())}"
            )
        text = (
            resources.files("marstrace")
            .joinpath("skills", skill, cls.SKILL_FILE)
            .read_text(encoding="utf-8")
        )
        if text.startswith(cls.FRONTMATTER_FENCE):
            end = text.find(
                "\n" + cls.FRONTMATTER_FENCE,
                len(cls.FRONTMATTER_FENCE),
            )
            if end >= 0:
                text = text[end + len(cls.FRONTMATTER_FENCE) + 1 :]
        return text.lstrip("\n")
