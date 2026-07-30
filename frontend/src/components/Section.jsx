function Section({ number, title, children, disabled }) {
  return (
    <section className={`section ${disabled ? "section-disabled" : ""}`}>
      <div className="section-header">
        <span className="section-number">{number}</span>
        <h2>{title}</h2>
      </div>
      <div className="section-body">{children}</div>
    </section>
  );
}

export default Section;