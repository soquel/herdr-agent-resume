import importlib.util
import json
import os
from io import StringIO
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("agent_resume", Path(__file__).parents[1] / "mods/agent-resume/agent_resume.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)

BOOT = "boot-now"
SID = "4f1c2a9e-7b3d-4e8a-9c51-2d6f0a8b3e17"
SID2 = "019a2b7c-3d4e-7f10-8a2b-c3d4e5f60718"


def label(agent="claude", session=SID, shell=100, boot=BOOT, at=1, cwd="/work", pid=200):
    return {"agent": agent, "session": session, "cwd": cwd, "transcript": f"/t/{session}.jsonl",
            "pid": pid, "shell": shell, "boot": boot, "at": at}


def terminal(address="t1", pid=300, fg_pid=None, fg_name="fish", lbl=None, state="live", where="api / 1"):
    labels = {"where": where}
    if lbl is not None:
        labels[ar.LABEL] = json.dumps(lbl)
    return {"address": address, "state": state, "pid": pid, "labels": labels,
            "foreground": {"pid": pid if fg_pid is None else fg_pid, "name": fg_name}}


def run_plan(terminals, boot=BOOT, folder_ok=lambda p: True, file_ok=lambda p: True, age=lambda pid: 60.0):
    return ar.plan(terminals, boot, folder_ok=folder_ok, file_ok=file_ok, age=age)


class LabelTests(unittest.TestCase):
    def test_reads_a_valid_label(self):
        self.assertEqual(ar.read_label({ar.LABEL: json.dumps(label())})["session"], SID)

    def test_ignores_broken_or_foreign_labels(self):
        for raw in ("not json", json.dumps([1]), json.dumps(label(agent="vim")),
                    json.dumps(label(session="x; rm -rf ~")), json.dumps(label(session="short"))):
            self.assertIsNone(ar.read_label({ar.LABEL: raw}), raw)
        self.assertIsNone(ar.read_label({}))
        self.assertIsNone(ar.read_label(None))


class LostTests(unittest.TestCase):
    def test_same_program_same_boot_is_not_lost(self):
        self.assertFalse(ar.lost({"pid": 100}, label(shell=100), BOOT))

    def test_another_program_is_lost(self):
        self.assertTrue(ar.lost({"pid": 101}, label(shell=100), BOOT))

    def test_another_boot_is_lost_even_with_a_reused_pid(self):
        self.assertTrue(ar.lost({"pid": 100}, label(shell=100, boot="old"), BOOT))

    def test_unknown_boot_falls_back_to_the_pid(self):
        self.assertFalse(ar.lost({"pid": 100}, label(shell=100, boot=None), None))
        self.assertTrue(ar.lost({"pid": None}, label(shell=100), BOOT))


class ShellTests(unittest.TestCase):
    def test_idle_shells(self):
        for name in ("fish", "-fish", "bash", "/bin/zsh", "sh"):
            self.assertTrue(ar.at_shell(terminal(fg_name=name)), name)

    def test_busy_terminals(self):
        self.assertFalse(ar.at_shell(terminal(fg_name="claude", fg_pid=301)))
        self.assertFalse(ar.at_shell(terminal(fg_name="bash", fg_pid=301)))  # a nested shell
        self.assertFalse(ar.at_shell(terminal(fg_name="vim")))
        self.assertFalse(ar.at_shell({"pid": 300}))
        self.assertFalse(ar.at_shell(terminal(pid=None)))


class CommandTests(unittest.TestCase):
    def test_claude_and_codex(self):
        self.assertEqual(ar.command(label(cwd="/a b/c")), f"cd '/a b/c' && claude --resume {SID}")
        self.assertEqual(ar.command(label(agent="codex", session=SID2)), f"cd '/work' && codex resume {SID2}")

    def test_grok_agy_pi_and_kiro(self):
        self.assertEqual(ar.command(label(agent="grok")), f"cd '/work' && grok --resume {SID}")
        self.assertEqual(ar.command(label(agent="agy")), f"cd '/work' && agy --conversation {SID}")
        self.assertEqual(ar.command(label(agent="pi")), f"cd '/work' && pi --session {SID}")
        self.assertTrue(ar.command(label(agent="kiro", session="sess_0123abcd")).endswith("launch-kiro sess_0123abcd"))
        self.assertTrue(ar.valid_session("custom.id_1"))

    def test_refuses_folders_that_cannot_be_typed_safely(self):
        for cwd in ("/it's", "/back\\slash", "/new\nline", "relative", "", None):
            self.assertIsNone(ar.command(label(cwd=cwd)), repr(cwd))


class FlagTests(unittest.TestCase):
    def test_keeps_claude_launch_flags_only(self):
        argv = ["--model", "opus", "--dangerously-skip-permissions", "--resume", SID, "fix the bug",
                "--add-dir", "/a", "/b", "--verbose", "-p", "--worktree", "w", "--permission-mode=plan", "--session-id", SID2]
        self.assertEqual(ar.launch_flags("claude", argv), [["--model", "opus"], ["--dangerously-skip-permissions"],
                                                           ["--add-dir", "/a", "/b"], ["--verbose"], ["--permission-mode=plan"]])

    def test_keeps_flags_codex_resume_accepts(self):
        argv = ["-m", "gpt-5", "-c", "model_reasoning_effort=high", "resume", SID2, "--full-auto",
                "-s", "workspace-write", "--model=o3", "-i", "shot.png", "--cd", "/x"]
        self.assertEqual(ar.launch_flags("codex", argv), [["-m", "gpt-5"], ["-c", "model_reasoning_effort=high"],
                                                          ["-s", "workspace-write"], ["--model=o3"]])

    def test_edge_cases(self):
        self.assertEqual(ar.launch_flags("claude", ["--model"]), [])                     # no value
        self.assertEqual(ar.launch_flags("claude", ["--model", "--verbose"]), [["--verbose"]])
        self.assertEqual(ar.launch_flags("claude", ["--add-dir", "--verbose"]), [["--verbose"]])
        self.assertEqual(ar.launch_flags("claude", ["--", "--model", "x"]), [])
        self.assertEqual(ar.launch_flags("claude", ["--model", "x" * 5000]), [])
        self.assertEqual(ar.launch_flags("claude", None), [])
        self.assertEqual(len(ar.flat(ar.launch_flags("claude", ["--add-dir", *["/d"] * 200]))), 0)

    def test_a_label_can_only_carry_kept_flags(self):
        raw = label()
        raw["args"] = ["--model", "x", "; rm -rf ~", "--evil", "--verbose"]
        self.assertEqual(ar.read_label({ar.LABEL: json.dumps(raw)})["args"], ["--model", "x", "--verbose"])
        raw["args"] = "--verbose"
        self.assertEqual(ar.read_label({ar.LABEL: json.dumps(raw)})["args"], [])

    def test_resume_command_with_flags(self):
        lbl = {**label(), "args": ["--model", "opus", "--append-system-prompt", "be brief", "--dangerously-skip-permissions"]}
        self.assertEqual(ar.command(lbl), f"cd '/work' && claude --resume {SID} --model opus "
                                          "--append-system-prompt 'be brief' --dangerously-skip-permissions")
        lbl["args"] = ["--append-system-prompt", "don't", "--model", "opus"]
        self.assertEqual(ar.command(lbl), f"cd '/work' && claude --resume {SID} --model opus")  # can't be typed: left out
        self.assertEqual(ar.resume_argv(lbl), ["claude", "--resume", SID, "--append-system-prompt", "don't", "--model", "opus"])
        codex = {**label(agent="codex", session=SID2), "args": ["-c", "a=b", "-s", "workspace-write"]}
        self.assertEqual(ar.command(codex), f"cd '/work' && codex resume {SID2} -c a=b -s workspace-write")

    def test_keeps_flags_the_other_resume_commands_accept(self):
        self.assertEqual(ar.launch_flags("grok", ["--resume", SID, "--model", "grok-4", "--always-approve",
                                                  "-c", "--permission-mode", "plan", "--worktree", "feat", "-p", "fix"]),
                         [["--model", "grok-4"], ["--always-approve"], ["--permission-mode", "plan"]])
        self.assertEqual(ar.launch_flags("agy", ["--conversation", SID, "--continue", "--model", "gemini",
                                                 "--dangerously-skip-permissions", "-p", "fix it"]),
                         [["--model", "gemini"], ["--dangerously-skip-permissions"]])
        self.assertEqual(ar.launch_flags("pi", ["--continue", "--model", "gpt-5", "--session", SID,
                                                "--thinking", "high", "--no-session", "-p", "hi"]),
                         [["--model", "gpt-5"], ["--thinking", "high"]])
        self.assertEqual(ar.launch_flags("kiro", ["chat", "--resume-id", SID, "--model", "claude",
                                                  "--trust-all-tools", "--no-interactive", "hello"]),
                         [["--model", "claude"], ["--trust-all-tools"]])


class FailureTests(unittest.TestCase):
    def test_last_words(self):
        self.assertEqual(ar.last_words("$ claude --resume x\nNo conversation found with session ID: x\n\nuser@host ~> \n"),
                         "No conversation found with session ID: x")
        self.assertIsNone(ar.last_words("user@host ~>"))
        self.assertIsNone(ar.last_words(""))

    def test_start_failed(self):
        back = terminal(pid=300, lbl=label(shell=100))
        self.assertTrue(ar.start_failed(back, ar.read_label(back["labels"]), BOOT))
        running = terminal(pid=300, fg_pid=301, fg_name="claude", lbl=label(shell=100))
        self.assertFalse(ar.start_failed(running, ar.read_label(running["labels"]), BOOT))
        noted = terminal(pid=300, lbl=label(shell=300))  # the hook noted it, then it was quit
        self.assertFalse(ar.start_failed(noted, ar.read_label(noted["labels"]), BOOT))

    def test_after(self):
        env = {"HC_TERMINAL_ID": "t1"}
        self.assertIsNone(ar.after("0", env=env))
        self.assertIsNone(ar.after("130", env=env))  # Ctrl+C: quit on purpose
        self.assertIsNone(ar.after("1", env={}))
        with tempfile.TemporaryDirectory() as run, mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": run}), \
             mock.patch.object(ar, "boot_id", return_value=BOOT):
            lost_term = terminal(pid=300, lbl=label(shell=100))
            with mock.patch.object(ar, "hc_json", return_value=lost_term):
                self.assertIsNone(ar.after("1", env=env))                 # no claim: `here` didn't start it
                Path(ar._claim_path("claude", SID)).write_text("999999999 1")
                self.assertIn("exited right away (code 1)", ar.after("1", env=env))
                self.assertFalse(os.path.exists(ar._claim_path("claude", SID)))
                Path(ar._claim_path("claude", SID)).write_text("999999999 1")
                old = time.time() - 60
                os.utime(ar._claim_path("claude", SID), (old, old))
                self.assertIsNone(ar.after("1", env=env))                 # ran a minute: not a failed resume
            Path(ar._claim_path("claude", SID)).write_text("999999999 1")
            with mock.patch.object(ar, "hc_json", return_value=terminal(pid=300, lbl=label(shell=300))):
                self.assertIsNone(ar.after("1", env=env))                 # the hook noted it


class PlanTests(unittest.TestCase):
    def test_resumes_a_lost_agent_at_an_idle_shell(self):
        todo, skipped = run_plan([terminal(lbl=label(shell=100))])
        self.assertEqual(skipped, [])
        self.assertEqual(todo, [{"terminal": "t1", "where": "api / 1", "agent": "claude", "session": SID,
                                 "cwd": "/work", "command": f"cd '/work' && claude --resume {SID}"}])

    def test_leaves_running_and_unlabeled_terminals_alone(self):
        todo, skipped = run_plan([terminal(lbl=label(shell=300)), terminal("t2"), terminal("t3", state="finished", lbl=label())])
        self.assertEqual((todo, skipped), ([], []))

    def test_skips_a_session_still_open_elsewhere(self):
        todo, skipped = run_plan([terminal("t1", lbl=label(shell=100)), terminal("t2", pid=400, lbl=label(shell=400))])
        self.assertEqual(todo, [])
        self.assertEqual([(s["terminal"], s["reason"]) for s in skipped], [("t1", "open in another terminal")])

    def test_resumes_one_session_once_newest_first(self):
        todo, skipped = run_plan([terminal("t1", lbl=label(shell=1, at=5)), terminal("t2", pid=301, lbl=label(shell=2, at=9))])
        self.assertEqual([t["terminal"] for t in todo], ["t2"])
        self.assertEqual([(s["terminal"], s["reason"]) for s in skipped], [("t1", "resumed in another terminal")])

    def test_reasons(self):
        cases = [
            (terminal(pid=None, lbl=label()), {}, ar.NOT_STARTED),
            (terminal(fg_name="vim", lbl=label()), {}, "something else is running there"),
            (terminal(lbl=label()), {"folder_ok": lambda p: False}, "its folder is gone"),
            (terminal(lbl=label()), {"file_ok": lambda p: False}, "no saved conversation to resume"),
            (terminal(lbl=label(cwd="/it's")), {}, "its folder name can't be typed safely"),
        ]
        for term, checks, reason in cases:
            todo, skipped = run_plan([term], **checks)
            self.assertEqual(todo, [])
            self.assertEqual([s["reason"] for s in skipped], [reason])

    def test_waits_for_a_restored_shell_still_starting(self):
        # Right after a core start, a restored shell runs its startup files before it is in front.
        for fg in ({"pid": None, "name": None}, {"pid": 301, "name": "starship"}):
            term = terminal(lbl=label())
            term["foreground"] = fg
            _, skipped = run_plan([term], age=lambda pid: 1.0)
            self.assertEqual([s["reason"] for s in skipped], [ar.NOT_STARTED], fg)
            _, skipped = run_plan([term], age=lambda pid: 30.0)
            self.assertEqual([s["reason"] for s in skipped], ["something else is running there"], fg)
            _, skipped = run_plan([term], age=lambda pid: None)
            self.assertEqual([s["reason"] for s in skipped], ["something else is running there"], fg)

    def test_a_young_shell_at_its_prompt_is_resumed(self):
        todo, skipped = run_plan([terminal(lbl=label())], age=lambda pid: 1.0)
        self.assertEqual(([t["terminal"] for t in todo], skipped), (["t1"], []))

    def test_claude_and_codex_with_the_same_id_are_different_sessions(self):
        todo, _ = run_plan([terminal("t1", lbl=label(shell=1)), terminal("t2", pid=301, lbl=label(agent="codex", shell=2))])
        self.assertEqual(sorted(t["agent"] for t in todo), ["claude", "codex"])


    def test_a_terminal_whose_shell_brought_the_agent_back(self):
        for fg in ({"name": "claude"}, {"name": "node-MainThread", "argv": ["node", "/usr/bin/codex"]},
                   {"name": "python3", "argv": ["python3", "/m/agent_resume.py", "here"]}):
            agent = "codex" if "codex" in str(fg) else "claude"
            term = terminal(lbl=label(agent=agent))
            term["foreground"] = {"pid": 301, **fg}
            _, skipped = run_plan([term])
            self.assertEqual([s["reason"] for s in skipped], [ar.BY_SHELL], fg)


class HereTests(unittest.TestCase):
    def decide(self, term, shell=300, **checks):
        return ar.decide_here(term, shell, BOOT, **{"folder_ok": lambda p: True, "file_ok": lambda p: True, **checks})

    def test_brings_back_a_lost_agent(self):
        found, why = self.decide(terminal(lbl=label(shell=100)))
        self.assertIsNone(why)
        self.assertEqual(found["session"], SID)
        found, why = self.decide(terminal(lbl=label(shell=300, boot="old")))
        self.assertIsNone(why)

    def test_leaves_everything_else(self):
        cases = [
            (terminal(lbl=label(shell=100)), {"shell": 301}, "not the terminal's own shell"),  # a nested shell
            (terminal(lbl=label(shell=100), state="finished"), {}, "not the terminal's own shell"),
            (terminal(), {}, "no session noted"),
            (terminal(lbl=label(shell=300)), {}, "agent not lost"),
            (terminal(lbl=label(shell=100)), {"folder_ok": lambda p: False}, "its folder is gone"),
            (terminal(lbl=label(shell=100)), {"file_ok": lambda p: False}, "no saved conversation to resume"),
        ]
        for term, extra, why in cases:
            self.assertEqual(self.decide(term, **extra), (None, why))

    def test_guards(self):
        self.assertEqual(ar.here(env={}), "not in a Herdr terminal")
        self.assertEqual(ar.here(env={"HC_TERMINAL_ID": "t1", "AGENT_RESUME_AUTO": "0"}), "turned off")

    def test_claims_a_session_once(self):
        with tempfile.TemporaryDirectory() as run, mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": run}):
            self.assertIsNone(ar.claim_holder("claude", SID))
            self.assertTrue(ar.claim("claude", SID))
            self.assertEqual(ar.claim_holder("claude", SID), os.getpid())
            self.assertFalse(ar.claim("claude", SID))
            self.assertTrue(ar.claim("codex", SID))
            Path(ar._claim_path("claude", SID)).write_text("999999999 1")  # a gone process
            self.assertIsNone(ar.claim_holder("claude", SID))
            self.assertTrue(ar.claim("claude", SID))


class ProcessTreeTests(unittest.TestCase):
    # 50 hook sh -> 40 claude -> 30 fish (the terminal) -> 20 herdr-core -> 1
    TREE = {50: (40, {"sh"}), 40: (30, {"claude"}), 30: (20, {"fish"}), 20: (1, {"herdr-core"}),
            # 90 claude -p -> 80 bash (a tool) -> 40 claude
            90: (80, {"claude"}), 80: (40, {"bash"}),
            # 70 codex -> 60 node (codex.js) -> 30 fish
            70: (60, {"codex"}), 60: (30, {"node"}),
            # 15 claude in tmux -> 10 tmux server -> 1
            15: (10, {"claude"}), 10: (1, {"tmux: server"}),
            # Kiro's own launcher: 110 helper -> bun -> kiro-cli-chat -> kiro-cli -> fish
            85: (30, {"kiro-cli"}), 95: (85, {"kiro-cli-chat"}), 100: (95, {"bun"}), 110: (100, {"kiro-cli-chat"}),
            # The same launcher under grok, which is a different agent.
            46: (30, {"grok"}), 86: (46, {"kiro-cli"}), 96: (86, {"kiro-cli-chat"}),
            101: (96, {"bun"}), 111: (101, {"kiro-cli-chat"})}

    def lookup(self, pid):
        if pid not in self.TREE:
            return None
        ppid, names = self.TREE[pid]
        return {"ppid": ppid, "names": names, "start": None}

    def test_finds_the_nearest_agent(self):
        self.assertEqual(ar.find_agent("claude", 50, self.lookup), 40)
        self.assertEqual(ar.find_agent("codex", 70, self.lookup), 70)
        self.assertIsNone(ar.find_agent("codex", 50, self.lookup))

    def test_placement(self):
        self.assertEqual(ar.placement(40, 30, self.lookup), "direct")
        self.assertEqual(ar.placement(70, 30, self.lookup), "direct")  # through codex's node wrapper
        self.assertEqual(ar.placement(90, 30, self.lookup), "nested")
        self.assertEqual(ar.placement(15, 30, self.lookup), "outside")
        self.assertEqual(ar.placement(30, 30, self.lookup), "direct")  # the agent is the terminal's program
        self.assertEqual(ar.placement(110, 30, self.lookup), "nested")  # parent kiro-cli counts as another agent
        self.assertEqual(ar.placement(ar.kiro_launcher(110, self.lookup), 30, self.lookup), "direct")
        self.assertEqual(ar.placement(ar.kiro_launcher(111, self.lookup), 30, self.lookup), "nested")  # grok is in between


class HookTests(unittest.TestCase):
    ENV = {"HC_TERMINAL_ID": "t7"}

    def payload(self, **extra):
        return json.dumps({"hook_event_name": "SessionStart", "session_id": SID, "transcript_path": "/t/x.jsonl",
                           "cwd": "/work", "source": "startup", **extra})

    def test_guards(self):
        self.assertEqual(ar.hook("claude", self.payload(), env={}), "not in a Herdr terminal")
        self.assertEqual(ar.hook("claude", "{", env=self.ENV), "input is not JSON")
        self.assertEqual(ar.hook("claude", self.payload(hook_event_name="Stop"), env=self.ENV), "not SessionStart")
        self.assertEqual(ar.hook("claude", self.payload(agent_id="sub"), env=self.ENV), "subagent")
        self.assertEqual(ar.hook("claude", self.payload(session_id="$(boom)"), env=self.ENV), "no usable session id")
        self.assertEqual(ar.hook("claude", self.payload(transcript_path=None), env=self.ENV),
                         "no transcript: the session can't be resumed")
        self.assertEqual(ar.hook("codex", self.payload(), env={**self.ENV, "CODEX_THREAD_ID": SID2}), "codex sub-thread")

    def test_saves_a_direct_session(self):
        with mock.patch.object(ar, "find_agent", return_value=40), \
             mock.patch.object(ar, "hc_json", return_value={"state": "live", "pid": 30}), \
             mock.patch.object(ar, "placement", return_value="direct"), \
             mock.patch.object(ar, "remember") as remember:
            self.assertEqual(ar.hook("codex", self.payload(), env={**self.ENV, "CODEX_THREAD_ID": SID}), "saved")
        remember.assert_called_once_with("t7", "codex", SID, "/work", "/t/x.jsonl", 40, 30)

    def saved(self, agent, raw, env=None):
        with mock.patch.object(ar, "find_agent", return_value=40), \
             mock.patch.object(ar, "hc_json", return_value={"state": "live", "pid": 30}), \
             mock.patch.object(ar, "placement", return_value="direct"), \
             mock.patch.object(ar, "remember") as remember:
            outcome = ar.hook(agent, raw, env={**self.ENV, **(env or {})})
        return outcome, remember

    def test_grok_agy_pi_and_kiro_payloads(self):
        outcome, remember = self.saved("grok", json.dumps({
            "hookEventName": "session_start", "sessionId": SID, "cwd": "/work", "transcriptPath": "/t/x.jsonl"}))
        self.assertEqual(outcome, "saved")
        remember.assert_called_once_with("t7", "grok", SID, "/work", "/t/x.jsonl", 40, 30)

        outcome, remember = self.saved("grok", json.dumps({"hook_event_name": "SessionStart", "cwd": "/work"}),
                                       env={"GROK_SESSION_ID": SID})
        self.assertEqual(outcome, "saved")
        self.assertTrue(remember.call_args[0][4].endswith(f"/.grok/sessions/%2Fwork/{SID}/updates.jsonl"))

        outcome, remember = self.saved("agy", json.dumps({
            "conversationId": SID, "workspacePaths": ["/work"], "transcriptPath": "/t/x.jsonl"}))
        self.assertEqual(outcome, "saved")
        remember.assert_called_once_with("t7", "agy", SID, "/work", "/t/x.jsonl", 40, 30)

        outcome, remember = self.saved("agy", json.dumps({
            "hookEventName": "PreInvocation", "conversationId": SID, "workspacePaths": ["/repo"],
            "transcriptPath": "/t/x.jsonl"}))
        self.assertEqual(outcome, "saved")
        self.assertEqual(remember.call_args[0][3], "/repo")

        outcome, remember = self.saved("pi", self.payload())
        self.assertEqual(outcome, "saved")
        remember.assert_called_once_with("t7", "pi", SID, "/work", "/t/x.jsonl", 40, 30)

        outcome, remember = self.saved("kiro", json.dumps({
            "hook_event_name": "SessionStart", "session_id": "sess_" + SID, "cwd": "/proj"}))
        self.assertEqual(outcome, "saved")
        self.assertEqual(remember.call_args[0][3], "/proj")
        self.assertTrue(remember.call_args[0][4].endswith(f"/sess_{SID}/session.json"))

        outcome, remember = self.saved("kiro", json.dumps({
            "hook_event_name": "UserPromptSubmit", "session_id": "sess_" + SID, "cwd": "/proj", "prompt": "hi"}))
        self.assertEqual(outcome, "saved")

        self.assertEqual(ar.hook("grok", json.dumps({"hook_event_name": "SessionStart", "session_id": SID,
                                                    "subagentType": "explore", "transcript_path": "/t/x.jsonl",
                                                    "cwd": "/work"}), env=self.ENV), "subagent")
        self.assertIsNone(ar.default_transcript("claude", SID, "/work"))
        self.assertIsNone(ar.default_transcript("pi", SID, "/work"))

    def test_a_repeat_of_the_same_session_does_not_note_it_again(self):
        lbl = label(agent="agy", shell=30, pid=40)
        terminal = {"state": "live", "pid": 30, "labels": {ar.LABEL: json.dumps(lbl)}}
        with mock.patch.object(ar, "find_agent", return_value=40), \
             mock.patch.object(ar, "hc_json", return_value=terminal), \
             mock.patch.object(ar, "placement", return_value="direct"), \
             mock.patch.object(ar, "remember") as remember:
            outcome = ar.hook("agy", json.dumps({"conversationId": SID, "workspacePaths": ["/work"],
                                                 "transcriptPath": lbl["transcript"]}), env=self.ENV)
        self.assertEqual(outcome, "already noted")
        remember.assert_not_called()

    def test_hook_stdout_stays_quiet_except_where_the_agent_requires_it(self):
        with mock.patch.object(ar, "hook", return_value="saved"), \
             mock.patch.object(ar, "log"), \
             mock.patch("sys.stdin", StringIO("{}")):
            for agent, expected in (("claude", ""), ("agy", "{}\n"), ("pi", "saved\n")):
                with mock.patch("sys.stdout", new_callable=StringIO) as out:
                    self.assertEqual(ar.main(["agent_resume.py", "hook", agent]), 0)
                self.assertEqual(out.getvalue(), expected, agent)

    def test_process_ancestry_preserves_outer_session(self):
        # Real find_agent/placement logic; only the OS and label write are mocked.
        for agent, name in (("claude", "claude"), ("codex", "codex"), ("kiro", "kiro-cli")):
            for nested in (False, True):
                tree = {30: {"ppid": 1, "names": {"fish"}},
                        40: {"ppid": 30, "names": {name}},
                        70: {"ppid": 40, "names": {"bash"}},
                        90: {"ppid": 70, "names": {name}},
                        100: {"ppid": 90 if nested else 40, "names": {"sh"}}}
                if agent == "kiro":
                    tree[100]["ppid"] = 110
                    tree[110] = {"ppid": 120, "names": {"kiro-cli-chat"}}
                    tree[120] = {"ppid": 130, "names": {"bun"}}
                    tree[130] = {"ppid": 90 if nested else 40, "names": {"kiro-cli-chat"}}
                terminal = {"state": "live", "pid": 30, "foreground": {"pid": 40, "name": name}}
                sid = "sess_" + SID if agent == "kiro" else SID
                with self.subTest(agent=agent, nested=nested), \
                     mock.patch.object(ar, "_proc_linux", side_effect=tree.get), \
                     mock.patch.object(ar.os.path, "isdir", return_value=True), \
                     mock.patch.object(ar.os, "getppid", return_value=100), \
                     mock.patch.object(ar, "hc_json", return_value=terminal), \
                     mock.patch.object(ar, "remember") as remember:
                    outcome = ar.hook(agent, self.payload(session_id=sid), env=self.ENV)
                if nested:
                    self.assertEqual(outcome, "agent runs nested")
                    remember.assert_not_called()
                else:
                    self.assertEqual(outcome, "saved")
                    self.assertEqual(remember.call_args.args[5], 40)

    def test_kiro_only_accepts_v3_sessions(self):
        self.assertEqual(ar.hook("kiro", self.payload(), env=self.ENV), "Kiro V2 is not supported")
        self.assertIsNone(ar.read_label({ar.LABEL: json.dumps(label(agent="kiro"))}))
        self.assertFalse(ar.is_session_event("kiro", {"hook_event_name": "agentSpawn"}))

    def test_kiro_metadata_location_and_forced_engine(self):
        with tempfile.TemporaryDirectory() as home:
            sid = "sess_" + SID
            predicted = Path(ar.default_transcript("kiro", sid, "/proj", home))
            self.assertEqual(predicted.name, "session.json")
            metadata = Path(home, ".kiro", "sessions", "multiple-roots", sid, "session.json")
            metadata.parent.mkdir(parents=True)
            metadata.write_text('{}')
            self.assertEqual(ar.default_transcript("kiro", sid, "/proj", home), str(metadata))
            saved = {**label(agent="kiro", session=sid), "args": ["--v2", "--agent-engine", "v2", "--mode", "spec"]}
            self.assertEqual(ar.resume_argv(saved), ["kiro-cli", "chat", "--v3", "--resume-id", sid, "--mode", "spec"])

    def test_skips_nested_agents(self):
        with mock.patch.object(ar, "find_agent", return_value=90), \
             mock.patch.object(ar, "hc_json", return_value={"state": "live", "pid": 30}), \
             mock.patch.object(ar, "placement", return_value="nested"), \
             mock.patch.object(ar, "remember") as remember:
            self.assertEqual(ar.hook("claude", self.payload(), env=self.ENV), "agent runs nested")
        remember.assert_not_called()


class KiroLifecycleTests(unittest.TestCase):
    def test_exit_without_a_prompt_forgets_session_and_failure_allows_retry(self):
        original = label(agent="kiro", session="sess_" + SID, shell=10, pid=11)
        current = {**original, "shell": 30, "pid": 40}
        terminal = {"pid": 30, "labels": {ar.LABEL: json.dumps(current)}}
        for code, elapsed, restore in ((0, 1, False), (130, 1, False), (1, 1, True), (1, 60, False)):
            with self.subTest(code=code, elapsed=elapsed), \
                 mock.patch.object(ar, "hc_json", return_value=terminal), \
                 mock.patch.object(ar, "hc") as hc:
                ar.kiro_finished("t7", original, 30, 40, code, elapsed)
                value = hc.call_args.args[0][-1]
                if restore:
                    self.assertEqual(json.loads(value.split("=", 1)[1]), original)
                else:
                    self.assertEqual(value, ar.LABEL + "=")

    def test_old_job_cannot_clear_a_replacement(self):
        original = label(agent="kiro", session="sess_" + SID)
        for shell, pid in ((30, 99), (99, 40)):
            terminal = {"pid": shell, "labels": {ar.LABEL: json.dumps({**original, "pid": pid})}}
            with mock.patch.object(ar, "hc_json", return_value=terminal), mock.patch.object(ar, "hc") as hc:
                ar.kiro_finished("t7", original, 30, 40, 0, 60)
                hc.assert_not_called()

    def test_wrapper_records_child_before_waiting_and_keeps_launch_flags(self):
        saved = {**label(agent="kiro", session="sess_" + SID), "args": ["--mode", "spec"]}
        def run(argv, cwd, started):
            self.assertIn("--v3", argv)
            started(40)
            return 0
        with mock.patch.object(ar, "run_as_job", side_effect=run), \
             mock.patch.object(ar, "remember") as remember, \
             mock.patch.object(ar, "kiro_finished") as finish:
            self.assertEqual(ar.run_kiro("t7", saved, 30), 0)
        self.assertEqual(remember.call_args.kwargs, {"watcher": False, "launch_args": ["--mode", "spec"]})
        self.assertEqual(finish.call_args.args[:5], ("t7", saved, 30, 40, 0))


class ClaudeRegistryTests(unittest.TestCase):
    def test_reads_a_running_session(self):
        with tempfile.TemporaryDirectory() as home:
            os.makedirs(os.path.join(home, ".claude", "sessions"))
            os.makedirs(os.path.join(home, ".claude", "projects", "-work"))
            Path(home, ".claude", "projects", "-work", f"{SID}.jsonl").write_text("{}\n")
            Path(home, ".claude", "sessions", "40.json").write_text(json.dumps(
                {"pid": 40, "sessionId": SID, "cwd": "/work", "kind": "interactive"}))
            self.assertEqual(ar.claude_session_of(40, home), (SID, "/work", os.path.join(home, ".claude", "projects", "-work", f"{SID}.jsonl")))
            self.assertIsNone(ar.claude_session_of(41, home))
            Path(home, ".claude", "sessions", "40.json").write_text(json.dumps(
                {"pid": 40, "sessionId": SID, "cwd": "/work", "kind": "background"}))
            self.assertIsNone(ar.claude_session_of(40, home))


if __name__ == "__main__":
    unittest.main()
