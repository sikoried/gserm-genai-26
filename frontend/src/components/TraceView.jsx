// Renders the agentic-RAG structured trace: which tool was called when, with
// what arguments and result, plus per-step and per-question token accounting.

function fmtArgs(args) {
  if (!args) return "";
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
    .join(", ");
}

const KIND_LABEL = { router: "route", tool: "tool", synthesis: "answer" };

export default function TraceView({ trace }) {
  if (!trace) return null;
  const { steps = [], stop_reason, elapsed_seconds, totals } = trace;
  const grand = totals?.total_tokens ?? 0;
  const toolCalls = steps.filter((s) => s.kind === "tool").length;

  return (
    <details className="trace-view">
      <summary>
        Trace · {toolCalls} tool{toolCalls === 1 ? "" : "s"} · {grand.toLocaleString()} tokens
        {typeof elapsed_seconds === "number" ? ` · ${elapsed_seconds.toFixed(2)}s` : ""}
        {stop_reason ? ` · stopped: ${stop_reason}` : ""}
      </summary>

      <table className="trace-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Step</th>
            <th>Detail</th>
            <th className="num">Tokens</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((s) => (
            <tr key={s.index} className={`trace-row trace-${s.kind}`}>
              <td>{s.index}</td>
              <td>
                <span className={`trace-kind trace-kind-${s.kind}`}>
                  {KIND_LABEL[s.kind] || s.kind}
                </span>
                {s.tool ? <span className="trace-tool"> {s.tool}</span> : null}
              </td>
              <td className="trace-detail">
                {s.kind === "tool" && s.arguments ? (
                  <code className="trace-args">({fmtArgs(s.arguments)})</code>
                ) : null}
                {s.result ? <span className="trace-result"> {s.result}</span> : null}
              </td>
              <td className="num">{(s.total_tokens ?? 0).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {totals ? (
        <div className="trace-totals">
          <span>router {totals.router?.total?.toLocaleString() ?? 0}</span>
          <span>tools {totals.tools?.total?.toLocaleString() ?? 0}</span>
          <span>synthesis {totals.synthesis?.total?.toLocaleString() ?? 0}</span>
          <span className="trace-grand">
            total {grand.toLocaleString()} ({(totals.prompt_tokens ?? 0).toLocaleString()} prompt
            {" + "}
            {(totals.completion_tokens ?? 0).toLocaleString()} completion)
          </span>
        </div>
      ) : null}
    </details>
  );
}
