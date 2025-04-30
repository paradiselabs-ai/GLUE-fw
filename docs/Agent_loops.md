

graph TD
    %% Main System / User Input
    UserInput["User/System Provides Main Goal"] --> LeadStart;

    %% Team Lead Orchestration Loop
    subgraph TeamLead [Team Lead Orchestration]
        direction TB
        LeadStart(["Lead Idle/Ready"]) --> GoalReceived;
        GoalReceived["Receive Main Goal/Task"] --> AnalyzeGoal["Analyze Goal & Plan Execution"];
        AnalyzeGoal --> InitWorkingState["Initialize Working Orchestration State"];
        InitWorkingState --> OrchestrationHub{"Check System State"};

        %% Delegation Path
        OrchestrationHub -- "Pending Task Available" --> SelectTask["Select Sub-task"];
        SelectTask --> IdentifyAndSelectMember["Identify & Select Member(s)"];
        IdentifyAndSelectMember -- Member Found --> DelegateTask["Delegate Sub-task to Member<br/>(Update Working State)"];
        %% Interaction Point: Lead -> Member Communication Channel
        DelegateTask -- Task Message --> CommChannel_Task[(Task Queue / Channel)];
        DelegateTask --> OrchestrationHub;

        %% No Member Path
        IdentifyAndSelectMember -- No Suitable Member --> HandleNoMember["Handle Inability to Delegate"];
        HandleNoMember --> ReportOutcomeLead["Prepare & Report Final Outcome (Failure)"];

        %% Report Processing Path
        OrchestrationHub -- "Report Received" --> ProcessReport["Process Member Report<br/>(Update Working State)"];
         %% Interaction Point: Member -> Lead Communication Channel
        CommChannel_Result[(Result Queue / Channel)] -- Result Report --> ProcessReport;
        ProcessReport --> EvaluateResult["Evaluate Member Result"];

        %% Evaluation Outcomes
        EvaluateResult -- Success --> MarkComplete["Mark Sub-task Complete<br/>(Update Working State)"];
        MarkComplete --> CheckAllComplete{"All Sub-tasks Done?"};
        CheckAllComplete -- Yes --> SynthesizeResult["Synthesize Final Result"];
        CheckAllComplete -- No --> OrchestrationHub;

        EvaluateResult -- Failure/Needs Correction --> HandleCorrection["Determine Correction Strategy"];
        HandleCorrection --> UpdateTaskForRetry["Update Sub-task State<br/>(Pending w/ Feedback)"];
        UpdateTaskForRetry --> OrchestrationHub;
        HandleCorrection -- "Strategy: Give Up/Fail Goal" --> ReportOutcomeLead;

        EvaluateResult -- Escalation Reported --> HandleEscalation["Handle Member Escalation"];
        HandleEscalation --> HandleCorrection;

        %% Other Paths
        OrchestrationHub -- "Timeout / Other Event" --> HandleOtherEvents["Handle Event"];
        HandleOtherEvents --> OrchestrationHub;

        %% Final Steps
        SynthesizeResult --> FinalPersistentUpdate["Optional: Update Persistent Memory"];
        FinalPersistentUpdate --> ReportOutcomeLead;
        SynthesizeResult --> ReportOutcomeLead;
        ReportOutcomeLead --> ClearWorkingState["Clear Working State"];
        ClearWorkingState --> LeadStart;
    end

    %% Team Member Execution Loop
    subgraph TeamMember [Team Member Execution]
        direction TB
        MemberStart(["Member Idle/Ready"]) --> TaskReceived;
         %% Interaction Point: Member <- Lead Communication Channel
        CommChannel_Task -- Task Message --> TaskReceived;
        TaskReceived["Task Received"] --> InitialAnalysis["Parse & Analyze Task"];
        InitialAnalysis --> ContextSetup["Gather Context"];
        ContextSetup --> PlanPhase["Plan Execution"];
        PlanPhase --> InitCounter["Init Turn Counter"];
        InitCounter --> ExecutionLoop{"Begin Execution Loop"};

        ExecutionLoop --> IncrementCounter["Inc Counter"];
        IncrementCounter --> AccessMemory["Access Memory"];
        AccessMemory --> SelectTool["Select Tool"];
        SelectTool --> InvokeTool["Invoke Tool"];
        InvokeTool --> GenResult["Get Result"];
        GenResult --> AutoSave["Auto Save Raw"];
        AutoSave --> MemoryDecision{"Save Curated?"};
        MemoryDecision -- Yes --> SaveToMemory["Use SaveTool"];
        MemoryDecision -- No --> SkipSave["Skip Save"];
        SaveToMemory --> StoreWM["Update WM"];
        SkipSave --> StoreWM;
        StoreWM --> SelfEval["Self-Evaluate"];

        SelfEval -- High Confidence --> IsComplete{"Task Done?"};
        SelfEval -- Medium Confidence --> Refine["Refine"];
        SelfEval -- Low Confidence --> ReviseStrategy["Revise"];
        SelfEval -- Critical Error --> Escalate["Prepare Escalation Report"];

        Refine --> ExecutionLoop;
        ReviseStrategy --> AttemptCounter{"Max Attempts?"};
        AttemptCounter -- No --> ExecutionLoop;
        AttemptCounter -- Yes --> PartialReport["Prepare Partial Report"];

        IsComplete -- Yes --> FormatResult["Format Success Report"];
        IsComplete -- No --> NextSubstep["Next Substep"];
        NextSubstep --> ExecutionLoop;

        %% Reporting Back
        FormatResult --> SendReport["Send Report to Lead"];
        PartialReport --> SendReport;
        Escalate --> SendReport;
        SendReport --> UpdateMemberMemory["Update Agent Memory"];
        UpdateMemberMemory --> MemberStart;
         %% Interaction Point: Member -> Lead Communication Channel
        SendReport -- Result Report --> CommChannel_Result;
    end

    %% Final System Output
    ReportOutcomeLead --> SystemOutput["Final Result/Status to User/System"];

    %% Styling Interaction Points
    linkStyle 0 stroke:#ff0000,stroke-width:2px,color:red;
    linkStyle 4 stroke:#0000ff,stroke-width:2px,color:blue;
    linkStyle 27 stroke:#0000ff,stroke-width:2px,color:blue;
    linkStyle 47 stroke:#ff0000,stroke-width:2px,color:red;








flowchart TD
    %% Entry Points
    Start(["Idle/Ready State"]) --> TaskReceived["Task Received from Team Lead"];
    TaskReceived --> InitialAnalysis["Parse & Analyze Task Requirements"];
   
    %% Context & Memory Integration
    InitialAnalysis --> ContextSetup["Gather Context:<br/>1. Task-specific context<br/>2. Relevant past task history<br/>3. System constraints"];
   
    %% Planning Phase
    ContextSetup --> PlanPhase["Plan Execution:<br/>1. Break into substeps<br/>2. Identify tool requirements<br/>3. Estimate confidence"];
    
    %% Initialize Turn Counter
    PlanPhase --> InitCounter["Initialize Turn Counter = 0"];
   
    %% Execution Cycle with Tool Selection and Execution
    InitCounter --> ExecutionLoop["Begin Execution Loop"];
    ExecutionLoop --> IncrementCounter["Increment Turn Counter"];
    IncrementCounter --> AccessMemory["Access:<br/>1. Full Curated Working Memory<br/>2. Last N Raw Outputs<br/>3. Current Turn Number"];
    AccessMemory --> SelectTool["Select Next Tool/Method"];
    SelectTool --> InvokeTool["Execute Tool/Method"];
   
    %% Results Processing with Working Memory
    InvokeTool --> GenResult["Initial Tool Result Given back"];
    GenResult --> AutoSave["Automatically Save to<br/>Raw Output History"];
    
    %% Memory Management Decision
    AutoSave --> MemoryDecision{"Is this worth<br/>saving to<br/>curated memory? [Prompted via its answer] "}
    MemoryDecision -- "Yes" --> SaveToMemory["Use SaveToMemory Tool<br/>to store validated findings,<br/>conclusions, or procedures"];
    MemoryDecision -- "No" --> SkipSave["Continue without<br/>explicit memory save"];
    SaveToMemory --> StoreWM["Update Working Memory"];
    SkipSave --> StoreWM;
   
    %% Enhanced Self-Evaluation
    StoreWM --> SelfEval["Self-Evaluate Result:<br/>1. Logical consistency<br/>2. Task alignment<br/>3. Constraint satisfaction<br/>4. Knowledge boundaries"];
   
    %% Decision Path
    SelfEval -- "High Confidence" --> IsComplete{"Is Task<br/>Fully Complete?"};
    SelfEval -- "Medium Confidence<br/>(Needs Refinement)" --> Refine["Refine Result<br/>(Keep Current Approach)"];
    SelfEval -- "Low Confidence<br/>(Needs New Approach)" --> ReviseStrategy["Revise Strategy/Tool"];
   
    %% Refinement Path
    Refine --> ExecutionLoop;
   
    %% Strategy Revision Path
    ReviseStrategy --> AttemptCounter{"Turn Counter >=<br/>Max Attempts?"};
    AttemptCounter -- "No" --> ExecutionLoop;
    AttemptCounter -- "Yes" --> PartialReport["Report Partial Success<br/>with Limitations"];
   
    %% Completion Paths
    IsComplete -- "Yes" --> FormatResult["Format Final Result<br/>with Supporting Context"];
    IsComplete -- "No" --> NextSubtask["Proceed to Next Substep"];
    NextSubtask --> ExecutionLoop;
   
    %% Reporting to Team Lead
    FormatResult --> ReportSuccess["Report Success to Team Lead<br/>with Complete Answer"];
    PartialReport --> UpdateSystemMemory["Update Agent Memory"];
    ReportSuccess --> UpdateSystemMemory;
   
    %% Return to Ready State
    UpdateSystemMemory --> Start;
   
    %% Progressive Escalation
    SelfEval -- "Critical Error<br/>Detected" --> Escalate["Escalate to Team Lead<br/>with Diagnostic Info"];
    Escalate --> UpdateSystemMemory;
   
    %% Style Nodes
    style Start fill:#d4f1f9,stroke:#333,stroke-width:2px
    style SelfEval fill:#ffe6b3,stroke:#333,stroke-width:2px
    style Escalate fill:#ffcccc,stroke:#333,stroke-width:2px
    style ReportSuccess fill:#ccffcc,stroke:#333,stroke-width:2px
    style ContextSetup fill:#e6ccff,stroke:#333,stroke-width:2px
    style PlanPhase fill:#e6ccff,stroke:#333,stroke-width:2px
    style StoreWM fill:#ffffcc,stroke:#333,stroke-width:1px
    style SelectTool fill:#d6eaf8,stroke:#333,stroke-width:1px
    style AccessMemory fill:#ffe6cc,stroke:#333,stroke-width:2px
    style AutoSave fill:#d5f5e3,stroke:#333,stroke-width:1px
    style SaveToMemory fill:#d5f5e3,stroke:#333,stroke-width:2px
    style MemoryDecision fill:#fdebd0,stroke:#333,stroke-width:1px
    style ExecutionLoop fill:#ebdef0,stroke:#333,stroke-width:1px











flowchart TD
    %% Entry Point
    Start(["Team Lead Idle/Ready State"]) --> GoalReceived["Receive Main Goal/Task"];

    %% Planning & State Initialization
    GoalReceived --> AnalyzeGoal["Analyze Goal & Plan Execution<br/>1. Understand requirements<br/>2. Break into logical Sub-tasks<br/>3. Identify dependencies"];
    AnalyzeGoal --> InitWorkingState["Initialize **Working Orchestration State**:<br/>- Sub-task list & status (Pending)<br/>- Member availability<br/>- Overall Goal Context"];

    %% Main Orchestration Loop Hub
    InitWorkingState --> OrchestrationHub{"Check System State:<br/>- Pending Tasks?<br/>- Incoming Reports?<br/>- Timeouts?"};

    %% Path 1: Delegate Pending Task
    OrchestrationHub -- "Pending Task Available" --> SelectTask["Select Highest Priority<br/>Pending Sub-task"];
    SelectTask --> IdentifyAndSelectMember["Identify Skills & Select<br/>Available Member(s)"];
    IdentifyAndSelectMember -- Member Found --> DelegateTask["Delegate Sub-task to Member<br/>(Update **Working State**: Task Assigned)"];
    %% Return to monitoring/checking state
    DelegateTask --> OrchestrationHub;

    %% Path 1.1: Handle No Suitable Member
    IdentifyAndSelectMember -- No Suitable Member --> HandleNoMember["Handle Inability to Delegate<br/>(Log Issue in **Working State**, Maybe Re-plan/Escalate?)"];
    %% Wait for state change (e.g., member free)
    HandleNoMember -- "Strategy: Wait/Retry" --> OrchestrationHub;
    HandleNoMember -- "Strategy: Escalate/Fail Goal" --> ReportOutcome["Prepare & Report<br/>Final Outcome (Failure)"];

    %% Path 2: Process Incoming Report
    OrchestrationHub -- "Report Received" --> ProcessReport["Process Member Report<br/>(Update **Working State**: Task Reported)"];
    ProcessReport --> EvaluateResult["Evaluate Member Result<br/>- Task Goal Met?<br/>- Quality Check<br/>- Errors/Escalations?"];

    %% Evaluation Outcomes
    EvaluateResult -- Success --> MarkComplete["Mark Sub-task Complete<br/>(Update **Working State**)"];
    MarkComplete --> CheckAllComplete{"All Necessary Sub-tasks<br/>Successfully Completed?"};
    CheckAllComplete -- Yes --> SynthesizeResult["Synthesize Final Result<br/>from Sub-task Outputs"];
    %% Continue orchestrating
    CheckAllComplete -- No --> OrchestrationHub;

    EvaluateResult -- Failure/Needs Correction --> HandleCorrection["Determine Correction Strategy<br/>(Retry? Re-assign? Modify?)<br/>(Check Sub-task Attempts)"];
    HandleCorrection -- "Strategy: Retry/Re-assign" --> UpdateTaskForRetry["Update Sub-task in **Working State**<br/>(e.g., Pending, Add Feedback)"];
    %% Loop back to potentially re-delegate
    UpdateTaskForRetry --> OrchestrationHub;
    HandleCorrection -- "Strategy: Give Up on Sub-task/Fail Goal" --> ReportOutcome;

    EvaluateResult -- Escalation Reported --> HandleEscalation["Handle Member Escalation<br/>(Log in **Working State**, Decide Action)"];
    %% Feed into correction/failure path
    HandleEscalation --> HandleCorrection;

    %% Path 3: Handle Timeouts or Other Events (Simplified)
    OrchestrationHub -- "Timeout / Other Event" --> HandleOtherEvents["Handle Event<br/>(e.g., Check long-running tasks, Escalate)"];
    HandleOtherEvents --> OrchestrationHub;

    %% Final Synthesis and Reporting
    SynthesizeResult --> FinalPersistentUpdate["Optional: Update **Persistent Memory**<br/>(Learnings, Final Summary)"];
    FinalPersistentUpdate --> ReportOutcome;
    %% If no persistent update needed
    SynthesizeResult --> ReportOutcome;

    ReportOutcome --> ClearWorkingState["Clear Task-Specific<br/>**Working Orchestration State**"];
    %% Return to Idle
    ClearWorkingState --> Start;

    %% Style Nodes
    style Start fill:#cce5ff,stroke:#333,stroke-width:2px
    style AnalyzeGoal fill:#d4eaff,stroke:#333,stroke-width:2px
    style OrchestrationHub fill:#e8daef,stroke:#333,stroke-width:2px
    style EvaluateResult fill:#ffe6b3,stroke:#333,stroke-width:2px
    style SynthesizeResult fill:#ccffcc,stroke:#333,stroke-width:2px
    style ReportOutcome fill:#ccffcc,stroke:#333,stroke-width:1px
    style DelegateTask fill:#d6eaf8,stroke:#333,stroke-width:1px
    style HandleCorrection fill:#ffcccc,stroke:#333,stroke-width:1px
    style HandleEscalation fill:#ffcccc,stroke:#333,stroke-width:1px
    style InitWorkingState fill:#e6ccff,stroke:#333,stroke-width:1px
    style ClearWorkingState fill:#e6ccff,stroke:#333,stroke-width:1px
    style FinalPersistentUpdate fill:#b3ffb3,stroke:#333,stroke-width:2px
