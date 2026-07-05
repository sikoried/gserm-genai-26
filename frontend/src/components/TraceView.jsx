// Renders the agentic-RAG structured trace: which tool was called when, with what
// arguments and result, which model produced each step, and per-step + per-question
// token accounting split into input / output / reasoning.

function fmtArgs(args) {
  if (!args) return "";
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
    .join(", ");
}

function shortModel(id) {
  if (!id) return "—";
  return id.split("/").pop();
}

const KIND_LABEL = { planner: "plan", router: "route", tool: "tool", synthesis: "answer" };

export default function TraceView({ trace }) {
  if (!trace) return null;
  const { steps = [], stop_reason, elapsed_seconds, totals, models } = trace;
  const grand = totals?.total_tokens ?? 0;
  const toolCalls = steps.filter((s) => s.kind === "tool").length;
  const multiHop = steps.some((s) => s.hop !== null && s.hop !== undefined);

  return (
    <details className="trace-view">
      <summary>
        Trace · {toolCalls} tool{toolCalls === 1 ? "" : "s"} · {grand.toLocaleString()} tokens
        {typeof elapsed_seconds === "number" ? ` · ${elapsed_seconds.toFixed(2)}s` : ""}
        {stop_reason ? ` · stopped: ${stop_reason}` : ""}
      </summary>

      {models && (models.router || models.synthesis) ? (
        <div className="trace-models">
          {models.router ? <span>router: <b>{shortModel(models.router)}</b></span> : null}
          {models.synthesis ? <span>synthesis: <b>{shortModel(models.synthesis)}</b></span> : null}
          {models.planner ? <span>planner: <b>{shortModel(models.planner)}</b></span> : null}
        </div>
      ) : null}

      <table className="trace-table">
        <thead>
          <tr>
            <th>#</th>
            {multiHop ? <th>Hop</th> : null}
            <th>Step</th>
            <th>Model</th>
            <th>Detail</th>
            <th className="num" title="input / output / reasoning">in/out/rsn</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((s) => (
            <tr key={s.index} className={`trace-row trace-${s.kind}`}>
              <td>{s.index}</td>
              {multiHop ? <td>{s.hop ?? "—"}</td> : null}
              <td>
                <span className={`trace-kind trace-kind-${s.kind}`}>
                  {KIND_LABEL[s.kind] || s.kind}
                </span>
                {s.tool ? <span className="trace-tool"> {s.tool}</span> : null}
              </td>
              <td className="trace-model">{shortModel(s.model_id)}</td>
              <td className="trace-detail">
                {s.kind === "tool" && s.arguments ? (
                  <code className="trace-args">({fmtArgs(s.arguments)})</code>
                ) : null}
                {s.result ? <span className="trace-result"> {s.result}</span> : null}
              </td>
              <td className="num">
                {(s.input_tokens ?? 0).toLocaleString()}/{(s.output_tokens ?? 0).toLocaleString()}/
                {(s.reasoning_tokens ?? 0).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {totals ? (
        <div className="trace-totals">
          <span>planner {totals.planner?.total?.toLocaleString() ?? 0}</span>
          <span>router {totals.router?.total?.toLocaleString() ?? 0}</span>
          <span>tools {totals.tools?.total?.toLocaleString() ?? 0}</span>
          <span>synthesis {totals.synthesis?.total?.toLocaleString() ?? 0}</span>
          <span className="trace-grand">
            total {grand.toLocaleString()} — input {(totals.input_tokens ?? 0).toLocaleString()},
            {" "}output {(totals.output_tokens ?? 0).toLocaleString()},
            {" "}reasoning {(totals.reasoning_tokens ?? 0).toLocaleString()}
          </span>
        </div>
      ) : null}
    </details>
  );
}
