# Mini RL Chatbot — Phase 1

A local Streamlit chatbot that learns which **response strategy** you prefer in a small number of contexts. Qwen 2.5 1.5B generates real answers through Ollama. A contextual bandit chooses **SHORT**, **DETAILED**, or **EXAMPLE**, and your explicit feedback updates its strategy estimates.

**This project does not fine-tune the language model.** It updates a small SQLite table of strategy scores, not Qwen's weights. There is no PPO, neural-network training, cloud LLM, or computer-control agent.

## 1. Architecture

```text
Your message
    │
    ▼
State builder: message-length bucket + broad intent
    │
    ▼
Contextual bandit ── reads saved Q-values ── SQLite
    │ chooses SHORT / DETAILED / EXAMPLE
    ▼
Prompt builder: strategy instruction + recent history + message
    │
    ▼
Local Ollama → Qwen 2.5 1.5B → generated answer
    │
    ▼
Streamlit displays the answer and four rating buttons
    │ only if you choose a rating
    ▼
Feedback manager ── one transaction ── SQLite
    ├─ store the rating and reward
    ├─ update the exact response's state-action Q-value, if eligible
    └─ record a learning event
```

The language model writes the answer. The bandit selects its style. The feedback manager supplies the reward. The database remembers what happened. Keeping these responsibilities in separate modules makes each one easier to understand and test.

### Project structure

```text
bot/
├── app.py                         # Streamlit UI and chat orchestration
├── styles.css                     # Monochrome chat layout and responsive styling
├── .streamlit/config.toml          # Native black-and-white Streamlit theme
├── config.py                      # Settings and validation
├── requirements.txt
├── README.md
├── .gitignore
├── chatbot/
│   ├── __init__.py
│   ├── llm.py                     # Local Ollama HTTP client and error handling
│   └── prompt_builder.py          # Strategy-specific prompts and bounded history
├── reinforcement_learning/
│   ├── __init__.py
│   ├── agent.py                   # Epsilon-greedy policy and Q update
│   ├── state.py                   # 12 coarse conversation contexts
│   └── reward.py                  # Explicit rating-to-reward mapping
├── feedback/
│   ├── __init__.py
│   └── feedback_manager.py        # Atomic, duplicate-safe feedback handling
├── database/
│   ├── __init__.py
│   ├── db.py                      # Connections, queries, conversations, resets
│   └── models.py                  # Data records, actions, and SQL schema
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Counts and observed reward averages
│   └── visualizations.py          # Matplotlib reward-over-time chart
├── tests/
│   ├── conftest.py                # Isolated temporary database fixtures
│   ├── test_agent.py
│   ├── test_feedback.py
│   ├── test_database.py
│   ├── test_llm.py
│   ├── test_metrics.py
│   └── test_app.py
└── data/
    ├── .gitkeep
    └── bot.sqlite3                # Created automatically when the app runs
```

The existing `.qodo/` folder is preserved. `.venv/`, Python caches, and SQLite files are local runtime artifacts, excluded by `.gitignore`.

## 2. Technologies in plain language

| Technology | What it does here |
| --- | --- |
| Python 3.11 | Runs the application and learning logic. |
| Streamlit | Turns Python code into a local web interface. A click reruns the script; session state remembers temporary UI choices. |
| Ollama | Runs a downloaded language model on your computer and exposes an HTTP API. |
| Qwen 2.5 1.5B | A small pretrained language model. “1.5B” describes roughly 1.5 billion parameters, not the amount of feedback needed. |
| SQLite | A database in a local file. It survives browser refreshes and application restarts. |
| Pandas | Groups stored interactions and computes evaluation statistics. |
| Matplotlib | Draws the average-reward chart. |
| Pytest | Runs repeatable checks on the policy, persistence, feedback, API handling, and UI. |

The Ollama client uses Python's standard-library HTTP tools. No PyTorch, Transformers, LangChain, external database server, or separate Ollama Python SDK is needed.

## 3. Installation

### Windows PowerShell

Open a terminal **inside the existing `bot` folder**. Do not create another project folder. On this machine that folder is `D:\bot`.

Python 3.11.9 is compatible. Check it with:

```powershell
py -3.11 --version
```

Create a virtual environment and install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A **virtual environment** keeps this project's Python packages separate from your other projects. Calling its Python executable directly avoids PowerShell activation-policy issues. If `.venv` is already set up, reuse it.

### macOS / Linux

Inside the existing project folder, with Python 3.11 installed:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 4. Install Ollama and download the model

Download Ollama from [ollama.com/download](https://ollama.com/download) and follow the instructions for your operating system. On Windows you can also use:

```powershell
winget install --id Ollama.Ollama --exact --source winget
```

Open a **new terminal** after installation so it sees the updated PATH. Start the Ollama desktop application, or run this in a separate terminal if its server is not already running:

```text
ollama serve
```

If it says port 11434 is already in use, Ollama may already be running. Check with `ollama list` instead of starting another server.

Download the exact model tag:

```text
ollama pull qwen2.5:1.5b
ollama list
```

The initial model download is about 1 GB; Ollama itself also needs disk space. CPU inference works, but speed and memory requirements depend on your computer. The first answer may take longer while the model loads.

Optional direct model check:

```text
ollama run qwen2.5:1.5b "Explain a Python list in one sentence."
```

Package and model downloads require internet access. **Chat inference stays local** at `http://127.0.0.1:11434`. This app accepts only loopback HTTP endpoints, rejects cloud-named/remote model entries, bypasses HTTP proxies, and does not follow redirects. It has no cloud fallback.

## 5. Start the application

From the project folder on Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

On macOS / Linux:

```bash
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501**. Keep Ollama and Streamlit running. Stop Streamlit with `Ctrl+C` in its terminal.

1. Check the model status in the sidebar.
2. Ask a question, such as “Why do Python lists use zero-based indexing?”
3. Read the generated answer. Its selected strategy is shown underneath.
4. Optionally click one feedback button. An unrated answer does not train the agent.
5. Open **Learning details** in the sidebar to see Q-values, action probabilities, strategy counts, and observed reward history.

The interface uses a quiet black-and-white theme, a centered welcome composer, and a bottom-docked message input during a conversation. Learning details open separately so the main view stays focused on chat.

Saved conversations can be reopened from **Your chats** in the sidebar. **New conversation** clears the chat context but shares the existing learning policy. **Reset learning**, inside sidebar **Settings**, starts a fresh learning run while retaining previous conversations, ratings, estimates, and events for inspection. Evaluation defaults to the current run; **Learning details** can include earlier runs.

### If Ollama is unavailable

The app remains usable for browsing history and inspecting the policy. It shows an actionable error rather than a fake model answer. Start Ollama, run the pull command if needed, then use **Check model connection** or **Retry response**. Failed requests do not count as chatbot responses, create rewards, or change Q-values. The unsent message stays available for retry in the current browser conversation; it is not saved to SQLite.

## 6. How the contextual bandit works

A **contextual bandit** chooses an action for the current situation, then learns from the immediate reward for that action. Unlike full reinforcement learning for a game, this version does not estimate long-term rewards over a sequence of future states. There is no next-state value or discount factor in its update.

### State: which broad situation are we in?

`reinforcement_learning/state.py` uses the current message only:

- **Length:** short (up to 80 characters), medium (81–240), or long (over 240).
- **Intent:** `how_to`, `explanation`, `question`, or `general`.

Ordered English keyword rules detect these intents. For example, “How do I sort a list?” becomes `short|how_to`; “Why is the sky blue?” becomes `short|explanation`. The rules check how-to requests first, then explanation requests, then questions, then the general fallback.

There are **3 × 4 = 12 states** and **12 × 3 = 36 possible state-action estimates** per learning run. The policy table contains only these coarse categories, not names, identities, or raw messages. The chat history itself is separately saved so conversations can be reopened.

### Actions: what style should the model use?

| ID | Action | Instruction sent to Qwen |
| --- | --- | --- |
| 0 | SHORT | Answer directly in one to three sentences. |
| 1 | DETAILED | Explain the reasoning in a few paragraphs or steps. |
| 2 | EXAMPLE | Explain with a concrete, appropriate example. |

The selected instruction appears in the **system message** sent to Ollama. The actions also have different output-token limits: 128, 384, and 320 respectively. A token is a piece of text used internally by the model; it is not exactly a word. Strategy selection therefore changes the real generation request.

Recent history is limited to 4,000 characters of complete messages, and a new message to 2,000 characters. The configured model context is 4,096 tokens. This is a lightweight size limit, not an exact tokenizer-based guarantee.

### Rewards: how was that specific response rated?

| Feedback | Reward |
| --- | ---: |
| Very helpful | +2 |
| Helpful | +1 |
| Neutral | 0 |
| Not helpful | -1 |

**No feedback is different from Neutral.** Neutral is an explicit rating and runs an update toward zero. Missing feedback creates no feedback row, no reward, and no learning event. Unknown labels are rejected.

### Policy: explore sometimes, otherwise use the best estimate

**Epsilon-greedy** means:

1. With probability **epsilon (ε)**, choose uniformly among all three strategies: **exploration**.
2. Otherwise, choose a strategy with the largest Q-value for this state: **exploitation**.
3. Break ties randomly, so an untrained agent does not always select SHORT.

Exploration can choose the current best action too. For three tied Q-values, all three strategies are equally likely even with epsilon set to zero.

Defaults:

```text
alpha = 0.2
initial epsilon = 0.2
minimum epsilon = 0.05
decay feedback count = 20
```

Exploration decays **per state**, based only on the number of feedback-driven updates in that state and the current run:

```text
epsilon(state) = max(0.05, 0.2 / (1 + updates_in_state / 20))
```

It is 0.20 before feedback, 0.10 after 20 updates, and 0.05 after 60. A rarely seen context keeps exploring even if other contexts have many ratings. Generating many unrated responses does not reduce epsilon.

When learning is paused, exploration is zero and Q-values are frozen. Ties still use random tie-breaking. Feedback received while paused is saved for evaluation and **never replayed later**. To learn, learning must have been enabled both when the response was selected and when feedback was submitted.

### Q-values and the update equation

`Q(s, a)` is an estimate of how rewarding strategy `a` is in state `s`. Every unseen pair starts at zero, represented implicitly until its first eligible feedback update.

```text
Q_new(s, a) = Q_old(s, a) + alpha × (reward − Q_old(s, a))
```

- `reward − Q_old` is the **prediction error**: how surprising the rating was.
- `alpha` is the **learning rate**: how far to move toward the new reward.
- Only the rated response's state-action pair changes.

Numerical example with `alpha = 0.2`, all for the same state and DETAILED action:

```text
Start:        Q = 0
Very helpful: Q = 0    + 0.2 × (2 − 0)    = 0.4
Helpful:      Q = 0.4  + 0.2 × (1 − 0.4)  = 0.52
Not helpful:  Q = 0.52 + 0.2 × (−1 − 0.52) = 0.216
```

A fixed alpha gives recent rewards continuing influence. This is an exponentially weighted estimate, **not** the arithmetic average shown in the evaluation table. If DETAILED has the largest estimate in this state, the agent will choose it more often during exploitation. A positive rating in one state does not automatically change other states.

The policy inspector exposes every state-action Q-value, update count, effective epsilon, and exact current selection probability. The database also stores the epsilon, selected estimate, and explore/exploit choice for each generated response.

## 7. Feedback reliability and storage

SQLite tables:

| Table | Contents |
| --- | --- |
| `conversations` | Conversation IDs and UTC creation times. |
| `messages` | User and assistant text, linked to a conversation. |
| `response_strategies` | Exact response/user-message IDs, action, state, model, selection details, and learning run. |
| `feedback` | One rating per response, its reward, and timestamp. |
| `state_action_values` | Q-value and update count for each learned state/action/run. |
| `learning_events` | One audit event per applied reward, with alpha, Q-before, and Q-after. |
| `learning_runs` | Initial run and subsequent resets. The newest run is active. |

A **primary key** uniquely identifies a row. A **foreign key** links it to another row. A **UNIQUE constraint** on `feedback.response_id` prevents two ratings for the same response. These database rules reinforce the application checks.

A **transaction** groups writes into an all-or-nothing operation. The feedback manager uses `BEGIN IMMEDIATE` to serialize SQLite writers before checking for a previous rating. The rating, Q update, and event are committed together, or all rolled back if something fails. Two simultaneous clicks cannot apply the reward twice. A second submission preserves the first rating rather than overwriting it.

Reset creates a new run instead of deleting history. An older response can still be rated, but its feedback cannot update the new policy. This also handles a reset occurring while an answer is being generated.

Data lives in `data/bot.sqlite3` by default, with SQLite's temporary `-wal` and `-shm` sidecars while active. All conversations share one policy; there are no user accounts. The database contains local, unencrypted chat text and feedback, not API keys or credentials. Close the app before making a simple file backup; copy the database after its active connections are closed.

## 8. Evaluation: is the strategy improving?

The app reports actual stored observations:

- Number of successfully generated responses.
- Number of feedback submissions.
- Average reward over **rated** responses only.
- Selection count, rating count, and mean reward per strategy.
- Cumulative average reward and the average of the last 10 ratings, ordered by **feedback time**.

The first rolling windows use the ratings available so far. Unrated answers are not filled with zero. Feedback collected while learning was paused is included in observed rewards but not in policy-update counts. A zero average is a measured value; an unavailable average means there are no ratings.

**An upward chart alone is not proof that learning helped.** Different questions, subjective ratings, a small sample, and only rating particularly good or bad answers can distort the result. Per-strategy averages also mix different contexts. This app does not automatically declare improvement.

A useful small manual evaluation:

1. Prepare a fixed set of questions covering several contexts and decide what “helpful” means before rating.
2. Collect a baseline in a separate database with learning paused. At zero Q-values, tie-breaking produces a uniform strategy baseline.
3. Train the main policy with genuine feedback on other questions.
4. Pause the trained policy and evaluate the same question set, with the same rating criteria and several repetitions. Compare average rewards, response counts, rating coverage, and results within comparable contexts.
5. Keep the same model and settings, and report both the observations and the sample size. Phase 1 does not calculate statistical significance or establish a causal improvement claim.

Use `BOT_DB_PATH` to separate baseline and learning data. Tests use temporary databases and mocked responses; their synthetic fixtures are **software checks, not evidence of chatbot improvement**, and never enter the application's learning history.

## 9. Configuration

Defaults live in `config.py`. Optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BOT_DB_PATH` | `data/bot.sqlite3` | Relative paths resolve from the project folder. |
| `BOT_OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint only. |
| `BOT_MODEL` | `qwen2.5:1.5b` | Downloaded local model name. |
| `BOT_ALPHA` | `0.2` | Learning rate; greater than 0 and at most 1. |
| `BOT_EPSILON` | `0.2` | Initial exploration probability. |
| `BOT_MIN_EPSILON` | `0.05` | Exploration floor, between 0 and initial epsilon. |
| `BOT_DECAY_FEEDBACK` | `20` | Per-state feedback count that halves initial epsilon, before the floor. |
| `BOT_TIMEOUT_SECONDS` | `180` | Inference request timeout. |

For example, in PowerShell before launching Streamlit:

```powershell
$env:BOT_ALPHA = "0.15"
$env:BOT_EPSILON = "0.10"
$env:BOT_MIN_EPSILON = "0.02"
$env:BOT_TIMEOUT_SECONDS = "300"
```

Restart Streamlit after changing environment variables. If you change the model or intent rules, start a new learning run to avoid mixing incompatible estimates. Configuring another local model changes generation, not the bandit algorithm.

## 10. Run the tests

Windows, from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

macOS / Linux:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

**Ollama is not required for the automated test suite.** Model API calls are mocked. Each test uses a temporary database, leaving real conversations and learning intact. Streamlit's built-in `AppTest` exercises chat submission, feedback controls, persistence, pause/reset behavior, and failure/retry handling without a browser.

Tests cover epsilon-greedy selection and tie-breaking, reward mapping, incremental updates, feedback-only learning, persistence, simultaneous duplicate submissions, transaction rollback, state generation, reset boundaries, database relationships, prompt strategies, model errors, and evaluation arithmetic.

### Phase 1 verification record

Verified on Windows with Python 3.11.9 on November 20, 2026:

- Complete automated suite: **98 passed in 8.49 seconds**. Temporary test files were directed outside the project with Pytest's `--basetemp` option.
- `python -m pip check`: **No broken requirements found.**
- Real local inference: Ollama 0.40.16 with downloaded `qwen2.5:1.5b`; all three strategy requests returned natural-language text.
- Unmocked Streamlit → bandit → Ollama → SQLite smoke test: one generated response, **zero ratings and zero learning updates**, using a temporary database.
- Running Streamlit server: `/` returned **HTTP 200**, and `/_stcore/health` returned **HTTP 200 / `ok`**. The smoke-test server was stopped afterward.

These checks establish application behavior, not answer accuracy or improvement from user feedback. Your regular `data/bot.sqlite3` is created on first use and starts without test ratings.

## 11. Current limitations and what to learn

- This learns **style preferences**, not new facts or language-model weights. Bad factual answers can still occur.
- Qwen 1.5B has limited reasoning capacity and can ignore a style instruction. Token limits may cut an answer short.
- State detection uses simple English rules. Other languages can be generated, but their intent classification is less reliable.
- Context is coarse and based on the current message. Follow-up questions may need context features this version does not use.
- The bounded history may omit older turns. Character limits do not exactly predict token usage in every language.
- All users and conversations on this local installation share one policy. Pause is a browser-session control; it does not pause another open session.
- Feedback is subjective and optional. Cold-start estimates need real ratings, and narrow usage may leave many contexts unlearned.
- This is a single-machine learning project, without authentication or multi-user deployment infrastructure. The startup command binds the UI to localhost.
- Real generation speed depends on hardware, and a timed-out request may still be finishing in Ollama. Retry deliberately.
- Evaluation is descriptive; it does not yet provide a controlled automatic benchmark or confidence intervals.

The main lessons are **separation of concerns**, **exploration versus exploitation**, **reward-driven incremental estimates**, **database transactions and idempotency** (the same rating cannot teach twice), **persistence across reruns**, and **testing behavior without a live language model**. Start by following one question from `app.py` through the state builder, strategy choice, Ollama prompt, and feedback transaction.
