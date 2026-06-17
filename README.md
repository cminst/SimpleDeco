$$\Large\textbf{Token-Level Control or Just a Better Mean?}$$
$$\textbf{Isolating the Gains of Adaptive Decoding}$$

---

**Abstract:** Adaptive decoding methods that predict per-token sampling parameters have
reported substantial gains over fixed decoding strategies. But what drives
these gains: genuine token-level control, or simply a better average
configuration? We develop an _inverse-temperature decomposition_ that
cleanly separates two mechanisms: shifting the global operating point and
performing residual token-level adaptation.

The theory shows that
teacher-forced log-probability is strictly concave in inverse temperature,
so any token-level variance incurs a quadratic curvature penalty that must
be overcome by precise gradient alignment. Testing this framework on
representative learned and heuristic controllers across three model
families and four benchmarks spanning reasoning and knowledge tasks
reveals a striking pattern: a fixed operating
point (obtained by simply averaging the learned controller's own
predictions) matches or exceeds the learned controller (AutoDeco) in the large
majority of evaluated settings.

Mechanistic diagnostics confirm the theory: residual temperature
variation exhibits near-chance gradient alignment and low Cauchy-Schwarz efficiency,
with curvature penalties dominating on the uncertain tokens where
adaptive control would matter most. A complementary analysis suggests
similar structural limits for adaptive top-p. These findings trace to
the limits of low-dimensional sampling control rather than to any one
controller. The decomposition toolkit and MeanShift baseline we
introduce give the field a sharper test of genuine token-level
adaptation.

---
