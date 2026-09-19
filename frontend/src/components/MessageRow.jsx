import RouteBadge from "./RouteBadge.jsx";
import ExamplesDrawer from "./ExamplesDrawer.jsx";

export default function MessageRow({ turn }) {
  const { incoming, result } = turn;

  return (
    <>
      <div className="row incoming">
        <div className="bubble">{incoming}</div>
      </div>
      <div className="row outgoing">
        <div className={`bubble${result.reply ? "" : " held-back"}`}>
          {result.reply || "No auto-reply sent — this one needs you personally."}
        </div>
        <div className="meta-line">
          <RouteBadge route={result.route} />
          <ExamplesDrawer examples={result.retrieved_examples} />
        </div>
      </div>
    </>
  );
}
