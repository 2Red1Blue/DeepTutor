# Connected Agents runtime contract

DeepTutor owns learning and tutoring state. Connected Agents are a native DeepTutor interaction surface; they are not Personal Runtime delegations or Agent Fabric operations. Every connection and streamed result identifies `execution_profile: native`, `runtime_owner: deeptutor`, the backend kind, and `managed_receipt: false`. A native run must not be projected as managed execution or used as proof that an external operation completed.

Local CLI backends execute with the DeepTutor server account's host credentials. Configured remote backends execute with deployment credentials. Administrators are deployment operators and may use enabled backends. Every ordinary account fails closed until an administrator assigns explicit backend ids in `grant.subagent_backends`; revoking the grant or disabling the backend blocks existing connections at the next invocation. Partner connections retain their separate per-partner grant and do not inherit deployment-backend grants. `GET /api/subagents/settings` returns only `consult_budget` and an empty backend map to an ordinary account; only an administrator receives the deployment backend configuration, including local execution flags, remote endpoint metadata, environment-variable names, profiles, and injected instructions.

Local CLI connections require an explicit working directory. The directory must exist and remain inside the caller's currently selected content workspace. Selecting a workspace is the owner approval; `DEEPTUTOR_WORKSPACE_ALLOWED_ROOTS` and a deployment-locked workspace constrain which roots an ordinary account can select. No Connected Agent falls back to the server process directory. Remote gateways and Partners do not receive a DeepTutor-local working directory.

## Multi-worker release gate

The persisted `(chat session, connection) → backend session` registry does not yet provide a distributed single-writer lease. A deployment with more than one API worker must explicitly set `enabled: false` for every deployment Connected Agent backend. Startup fails if `backend_workers > 1` and any configurable backend remains enabled, including an omitted backend whose safe local default is enabled. Invocation performs the same worker-count fence, including Partner execution, so bypassing the supported launcher cannot resume a native backend from two workers. Redis turn coordination does not satisfy this separate lease requirement.

The release gate remains open until an acceptance test starts two worker processes, concurrently resumes the same connection, proves that exactly one owns the backend session, kills that owner, and proves that a fenced successor can reconcile before resuming. An in-process lock is not sufficient and must not be presented as multi-worker support.

Every backend event and every terminal `ToolResult` produced after execution resolution, including backend exceptions, empty answers, and backend-declared failures, retains the native provenance fields. Resolution failures also derive the same marker from the requested backend kind. Failure is not evidence of managed execution and must not drop the runtime owner marker.

## Optional ecosystem integration

A future managed learning profile may admit a DeepTutor-owned learning target to Personal Runtime and execute it through Agent Fabric. That adapter belongs in this repository and must be explicit; it does not replace ordinary tutoring, Connected Agents, sessions, scheduling, or recovery. Workbench may project a public DeepTutor read model and deep-link back to the native UI. Observability may record spans but never owns learning or execution truth.
