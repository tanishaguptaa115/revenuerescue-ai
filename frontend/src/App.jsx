import { useEffect, useState } from "react";
import "./App.css";

const MERCHANT_ID = "985d35c9-fdbb-4d2a-b974-a71c72b86fae";

const initialForm = {
  customerId: "cust_000000",
  amount: 1061.65,
  paymentMethod: "UPI",
  failureReason: "insufficient_funds",
  retryCount: 1,
  isSoftFailure: true,
  numPaymentMethodsUsedRecently: 1,
  ipCountryMismatch: false,
  deviceChangeFlag: false,
  velocityTxnCount1h: 1,
  velocityTxnCount24h: 2,
  daysSinceLastSuccessfulPayment: 5,
};

const scenarios = {
  safe: {
    label: "Safe Recovery",
    description: "Low risk, recoverable failure",
    values: {
      customerId: "cust_000010",
      amount: 850,
      paymentMethod: "UPI",
      failureReason: "insufficient_funds",
      retryCount: 1,
      isSoftFailure: true,
      numPaymentMethodsUsedRecently: 1,
      ipCountryMismatch: false,
      deviceChangeFlag: false,
      velocityTxnCount1h: 1,
      velocityTxnCount24h: 2,
      daysSinceLastSuccessfulPayment: 3,
    },
  },

  highRisk: {
    label: "High Risk",
    description: "Verified high-risk transaction",
    values: {
      customerId: "cust_002147",
      amount: 9008.91,
      paymentMethod: "UPI",
      failureReason: "issuer_declined",
      retryCount: 0,
      isSoftFailure: false,
      numPaymentMethodsUsedRecently: 2,
      ipCountryMismatch: true,
      deviceChangeFlag: true,
      velocityTxnCount1h: 1,
      velocityTxnCount24h: 1,
      daysSinceLastSuccessfulPayment: "",
    },
  },

  fatigue: {
    label: "Retry Fatigue",
    description: "Recovery blocked after repeated retries",
    values: {
      customerId: "cust_000200",
      amount: 1250,
      paymentMethod: "UPI",
      failureReason: "insufficient_funds",
      retryCount: 3,
      isSoftFailure: true,
      numPaymentMethodsUsedRecently: 2,
      ipCountryMismatch: false,
      deviceChangeFlag: false,
      velocityTxnCount1h: 1,
      velocityTxnCount24h: 4,
      daysSinceLastSuccessfulPayment: 8,
    },
  },

  suspicious: {
    label: "Suspicious",
    description: "Verified fraudulent transaction",
    values: {
      customerId: "cust_002693",
      amount: 522.91,
      paymentMethod: "Wallet",
      failureReason: "card_declined_issuer",
      retryCount: 0,
      isSoftFailure: false,
      numPaymentMethodsUsedRecently: 2,
      ipCountryMismatch: true,
      deviceChangeFlag: true,
      velocityTxnCount1h: 0,
      velocityTxnCount24h: 0,
      daysSinceLastSuccessfulPayment: "",
    },
  },
};

function getDecisionExplanation(result) {
  if (!result) return null;

  const policy = result.metadata?.merchant_policy;
  const risk = Number(result.risk_score ?? 0);
  const recovery = Number(result.recovery_score ?? 0);

  const riskTolerance = Number(
    result.merchant_risk_tolerance ?? 0
  );

  const recoveryThreshold = Number(
    result.metadata?.recovery_score_threshold ?? 0
  );

  const retryCount = Number(
    result.metadata?.retry_count_so_far ?? 0
  );

  if (
    policy &&
    retryCount >= Number(policy.max_auto_retries)
  ) {
    return `This payment has already reached the merchant's maximum automated retry limit of ${policy.max_auto_retries}. Further recovery attempts are suppressed to avoid repeated payment attempts.`;
  }

  switch (result.reason_code) {
    case "LOW_RECOVERY_PROBABILITY":
      return `Recovery probability is ${(recovery * 100).toFixed(
        1
      )}%, below the ${(recoveryThreshold * 100).toFixed(
        0
      )}% merchant auto-recovery threshold. Risk remains below the ${(riskTolerance * 100).toFixed(
        0
      )}% review threshold, so no automated recovery is attempted.`;

    case "MAX_RETRIES_REACHED":
      return `Automated recovery is suppressed because the transaction has already reached ${retryCount} retries, reducing the risk of repeated payment attempts.`;

    case "HIGH_RISK_REVIEW":
      return `Risk is ${(risk * 100).toFixed(
        1
      )}%, at or above the ${(riskTolerance * 100).toFixed(
        0
      )}% review threshold, so the transaction is routed for review.`;

    case "HIGH_RISK_BLOCK":
      return `Risk is ${(risk * 100).toFixed(
        1
      )}%, exceeding the merchant's automated safety boundary. The transaction is blocked to protect the merchant.`;

    case "RECOVERY_RECOMMENDED":
      return `Recovery probability is ${(recovery * 100).toFixed(
        1
      )}%, above the ${(recoveryThreshold * 100).toFixed(
        0
      )}% merchant threshold, while risk remains within the merchant's allowed range. Automated recovery is recommended.`;

    case "HARD_FAILURE_NO_RECOVERY":
      return "This payment failure is classified as a hard failure, so automated recovery is intentionally skipped.";

    default:
      return result.human_readable_reason;
  }
}

function getTimelineSteps(result, customerProfile) {
  if (!result) return [];

  return [
    {
      label: "Transaction",
      detail: "Received",
      status: "complete",
    },
    {
      label: "Customer",
      detail: customerProfile
        ? "Enriched"
        : "Profile unavailable",
      status: customerProfile
        ? "complete"
        : "warning",
    },
    {
      label: "Risk Model",
      detail:
        typeof result.risk_score === "number"
          ? "Scored"
          : "Pending",
      status:
        typeof result.risk_score === "number"
          ? "complete"
          : "pending",
    },
    {
      label: "Recovery Model",
      detail:
        typeof result.recovery_score === "number"
          ? "Scored"
          : "Skipped",
      status:
        typeof result.recovery_score === "number"
          ? "complete"
          : "skipped",
    },
    {
      label: "Merchant Policy",
      detail: result.metadata?.merchant_policy
        ? "Applied"
        : "Unavailable",
      status: result.metadata?.merchant_policy
        ? "complete"
        : "warning",
    },
    {
      label: "Decision",
      detail: result.action || "Pending",
      status: result.action
        ? "complete"
        : "pending",
    },
    {
      label: "Supabase Audit",
      detail: result.metadata?.persistence
        ? "Saved"
        : "Pending",
      status: result.metadata?.persistence
        ? "complete"
        : "pending",
    },
  ];
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState("");

  const [customerProfile, setCustomerProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState("");

  const update = (key, value) => {
    setForm((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  useEffect(() => {
    const customerId = form.customerId.trim();

    if (!customerId) {
      setCustomerProfile(null);
      setProfileError("");
      return;
    }

    let cancelled = false;

    const loadCustomerProfile = async () => {
      setProfileLoading(true);
      setProfileError("");

      try {
        const response = await fetch(
          `http://127.0.0.1:8000/customer-profile/${encodeURIComponent(
            customerId
          )}`
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
            "Unable to load customer profile."
          );
        }

        if (!cancelled) {
          setCustomerProfile(data);
        }
      } catch (err) {
        if (!cancelled) {
          setCustomerProfile(null);
          setProfileError(
            err.message ||
            "Unable to load customer profile."
          );
        }
      } finally {
        if (!cancelled) {
          setProfileLoading(false);
        }
      }
    };

    loadCustomerProfile();

    return () => {
      cancelled = true;
    };
  }, [form.customerId]);

  const applyScenario = (scenarioKey) => {
    setForm(scenarios[scenarioKey].values);
    setResult(null);
    setError("");
    setCopied("");
  };

  const resetDemo = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
    setCopied("");
  };

  const copyValue = async (value, label) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(label);

      setTimeout(() => {
        setCopied("");
      }, 1600);
    } catch {
      setError("Unable to copy this ID.");
    }
  };

  const analyzeTransaction = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    const rawDays =
      form.daysSinceLastSuccessfulPayment;

    const daysSinceSuccess =
      rawDays === "" ? null : Number(rawDays);

    const hasPriorSuccess =
      daysSinceSuccess !== null &&
      Number.isFinite(daysSinceSuccess) &&
      daysSinceSuccess >= 0;

    const payload = {
      merchant_id: MERCHANT_ID,
      customer_id: form.customerId.trim(),
      amount: Number(form.amount),
      currency: "INR",
      payment_method: form.paymentMethod,
      payment_failed: true,
      failure_reason: form.failureReason,
      is_soft_failure: form.isSoftFailure,
      retry_count_so_far: Number(form.retryCount),

      risk_features: {
        amount: Number(form.amount),
        payment_method: form.paymentMethod,
        num_payment_methods_used_recently:
          Number(form.numPaymentMethodsUsedRecently),
        ip_country_mismatch: form.ipCountryMismatch,
        device_change_flag: form.deviceChangeFlag,
        velocity_txn_count_1h:
          Number(form.velocityTxnCount1h),
        velocity_txn_count_24h:
          Number(form.velocityTxnCount24h),
        days_since_last_successful_payment:
          daysSinceSuccess,
        has_prior_success: hasPriorSuccess,
      },

      recovery_features: {
        amount: Number(form.amount),
        payment_method: form.paymentMethod,
        failure_reason_code: form.failureReason,
        is_soft_failure: form.isSoftFailure,
        retry_count_so_far:
          Number(form.retryCount),
        days_since_last_successful_payment:
          daysSinceSuccess,
        has_prior_success: hasPriorSuccess,
      },
    };

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/score-and-decide",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Request failed"
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
        "Unable to connect to the RevenueRescue API."
      );
    } finally {
      setLoading(false);
    }
  };

  const riskPercent = result
    ? Number(result.risk_score * 100).toFixed(1)
    : "—";

  const recoveryPercent = result
    ? Number(result.recovery_score * 100).toFixed(1)
    : "—";

  const riskWidth = result
    ? Math.min(result.risk_score * 100, 100)
    : 0;

  const recoveryWidth = result
    ? Math.min(result.recovery_score * 100, 100)
    : 0;

  const decisionClass =
    result?.action?.toLowerCase() || "";

  const policy =
    result?.metadata?.merchant_policy;

  const persistence =
    result?.metadata?.persistence;

  const decisionExplanation =
    getDecisionExplanation(result);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">R</span>

          <div>
            <h1>RevenueRescue AI</h1>
            <p>
              Autonomous payment recovery & risk
              intelligence
            </p>
          </div>
        </div>

        <div className="topbar-right">
          <div className="engine-status">
            <span className="status-dot"></span>
            AI Engine Online
          </div>

          <div className="api-label">
            FASTAPI · SUPABASE · ML
          </div>
        </div>
      </header>

      <main className="dashboard">
        <section className="hero-copy">
          <div className="hero-topline">
            <span className="eyebrow">
              MERCHANT CONTROL CENTER
            </span>

            <button
              className="reset-btn"
              onClick={resetDemo}
              type="button"
            >
              ↻ Reset Demo
            </button>
          </div>

          <h2>
            Recover revenue without{" "}
            <span>ignoring risk.</span>
          </h2>

          <p>
            RevenueRescue AI combines customer context,
            trained ML models, and merchant policy to
            decide what should happen after a payment
            failure.
          </p>
        </section>

        <section className="kpi-strip">
          <div className="kpi-card">
            <span>RISK MODEL</span>

            <strong>
              {result ? `${riskPercent}%` : "—"}
            </strong>

            <small>fraud probability</small>
          </div>

          <div className="kpi-card">
            <span>RECOVERY MODEL</span>

            <strong>
              {result
                ? `${recoveryPercent}%`
                : "—"}
            </strong>

            <small>recovery probability</small>
          </div>

          <div className="kpi-card">
            <span>DECISION</span>

            <strong className={decisionClass}>
              {result?.action || "READY"}
            </strong>

            <small>
              {result
                ? "policy-aware outcome"
                : "awaiting analysis"}
            </small>
          </div>

          <div className="kpi-card">
            <span>CUSTOMER</span>

            <strong className="customer-kpi">
              {form.customerId}
            </strong>

            <small>profile-backed context</small>
          </div>
        </section>

        <section className="scenario-section">
          <div className="section-label-row">
            <div>
              <span className="section-kicker">
                DEMO SCENARIOS
              </span>

              <h3>
                Show the decision engine in action
              </h3>
            </div>

            <span className="section-help">
              One click changes transaction signals
            </span>
          </div>

          <div className="scenario-grid">
            {Object.entries(scenarios).map(
              ([key, scenario]) => (
                <button
                  key={key}
                  className="scenario-card"
                  type="button"
                  onClick={() =>
                    applyScenario(key)
                  }
                >
                  <strong>
                    {scenario.label}
                  </strong>

                  <span>
                    {scenario.description}
                  </span>
                </button>
              )
            )}
          </div>
        </section>

        <section className="main-grid">
          <div className="card input-card">
            <div className="card-heading">
              <div>
                <span className="step">01</span>
                <h3>Transaction Signals</h3>
              </div>

              <span className="small-label">
                INPUT
              </span>
            </div>

            <div className="form-grid">
              <label>
                Customer ID

                <input
                  type="text"
                  value={form.customerId}
                  onChange={(e) =>
                    update(
                      "customerId",
                      e.target.value
                    )
                  }
                  placeholder="cust_000000"
                />
              </label>

              <label>
                Amount

                <div className="input-wrap">
                  <span>₹</span>

                  <input
                    type="number"
                    min="1"
                    step="0.01"
                    value={form.amount}
                    onChange={(e) =>
                      update(
                        "amount",
                        e.target.value
                      )
                    }
                  />
                </div>
              </label>

              <label>
                Payment Method

                <select
                  value={form.paymentMethod}
                  onChange={(e) =>
                    update(
                      "paymentMethod",
                      e.target.value
                    )
                  }
                >
                  <option>UPI</option>
                  <option>Card</option>
                  <option>Net Banking</option>
                  <option>Wallet</option>
                </select>
              </label>

              <label>
                Failure Reason

                <select
                  value={form.failureReason}
                  onChange={(e) =>
                    update(
                      "failureReason",
                      e.target.value
                    )
                  }
                >
                  <option value="insufficient_funds">
                    Insufficient Funds
                  </option>

                  <option value="issuer_declined">
                    Issuer Declined
                  </option>

                  <option value="network_error">
                    Network Error
                  </option>

                  <option value="authentication_failed">
                    Authentication Failed
                  </option>

                  <option value="card_declined_issuer">
                    Card Declined by Issuer
                  </option>
                </select>
              </label>

              <label>
                Retry Count

                <input
                  type="number"
                  min="0"
                  value={form.retryCount}
                  onChange={(e) =>
                    update(
                      "retryCount",
                      e.target.value
                    )
                  }
                />
              </label>

              <label>
                Velocity · 1h

                <input
                  type="number"
                  min="0"
                  value={form.velocityTxnCount1h}
                  onChange={(e) =>
                    update(
                      "velocityTxnCount1h",
                      e.target.value
                    )
                  }
                />
              </label>

              <label>
                Velocity · 24h

                <input
                  type="number"
                  min="0"
                  value={form.velocityTxnCount24h}
                  onChange={(e) =>
                    update(
                      "velocityTxnCount24h",
                      e.target.value
                    )
                  }
                />
              </label>

              <label>
                Recent Payment Methods

                <input
                  type="number"
                  min="0"
                  value={
                    form.numPaymentMethodsUsedRecently
                  }
                  onChange={(e) =>
                    update(
                      "numPaymentMethodsUsedRecently",
                      e.target.value
                    )
                  }
                />
              </label>

              <label>
                Days Since Last Success

                <input
                  type="number"
                  min="0"
                  value={
                    form.daysSinceLastSuccessfulPayment
                  }
                  placeholder="Unknown"
                  onChange={(e) =>
                    update(
                      "daysSinceLastSuccessfulPayment",
                      e.target.value
                    )
                  }
                />
              </label>
            </div>

            <div className="signal-grid">
              <label className="signal-card">
                <input
                  type="checkbox"
                  checked={form.isSoftFailure}
                  onChange={(e) =>
                    update(
                      "isSoftFailure",
                      e.target.checked
                    )
                  }
                />

                <span>
                  <strong>Soft Failure</strong>
                  <small>
                    Eligible recovery signal
                  </small>
                </span>
              </label>

              <label className="signal-card">
                <input
                  type="checkbox"
                  checked={
                    form.ipCountryMismatch
                  }
                  onChange={(e) =>
                    update(
                      "ipCountryMismatch",
                      e.target.checked
                    )
                  }
                />

                <span>
                  <strong>
                    Country Mismatch
                  </strong>

                  <small>
                    IP differs from profile
                  </small>
                </span>
              </label>

              <label className="signal-card">
                <input
                  type="checkbox"
                  checked={form.deviceChangeFlag}
                  onChange={(e) =>
                    update(
                      "deviceChangeFlag",
                      e.target.checked
                    )
                  }
                />

                <span>
                  <strong>Device Changed</strong>

                  <small>
                    Recent device change
                  </small>
                </span>
              </label>
            </div>

            <div className="intelligence-panel">
              <div className="intelligence-header">
                <div className="intelligence-title">
                  <div className="intel-icon">
                    ✦
                  </div>

                  <div>
                    <strong>
                      Customer Intelligence
                    </strong>

                    <p>
                      Historical profile used for ML
                      enrichment
                    </p>
                  </div>
                </div>

                {profileLoading ? (
                  <span className="profile-status loading">
                    LOADING
                  </span>
                ) : customerProfile ? (
                  <span className="profile-status connected">
                    ● CONNECTED
                  </span>
                ) : (
                  <span className="profile-status error">
                    UNAVAILABLE
                  </span>
                )}
              </div>

              {profileLoading ? (
                <div className="profile-loading">
                  Loading customer profile...
                </div>
              ) : customerProfile ? (
                <div className="profile-grid">
                  <div className="profile-item">
                    <span>PROFILE</span>

                    <strong>
                      {customerProfile.archetype.replaceAll(
                        "_",
                        " "
                      )}
                    </strong>
                  </div>

                  <div className="profile-item">
                    <span>
                      AVG TRANSACTION
                    </span>

                    <strong>
                      ₹
                      {Number(
                        customerProfile
                          .avg_transaction_amount_customer
                      ).toLocaleString(
                        "en-IN",
                        {
                          maximumFractionDigits: 2,
                        }
                      )}
                    </strong>
                  </div>

                  <div className="profile-item">
                    <span>PAST SUCCESS</span>

                    <strong>
                      {(
                        customerProfile
                          .customer_past_success_rate *
                        100
                      ).toFixed(1)}
                      %
                    </strong>
                  </div>

                  <div className="profile-item">
                    <span>PAST RECOVERY</span>

                    <strong>
                      {(
                        customerProfile
                          .customer_past_recovery_rate *
                        100
                      ).toFixed(1)}
                      %
                    </strong>
                  </div>

                  <div className="profile-item">
                    <span>CHARGEBACKS</span>

                    <strong>
                      {
                        customerProfile
                          .chargeback_history_count
                      }
                    </strong>
                  </div>

                  <div className="profile-item">
                    <span>TENURE</span>

                    <strong>
                      {
                        customerProfile.customer_tenure_days
                      }{" "}
                      days
                    </strong>
                  </div>
                </div>
              ) : (
                <div className="profile-error">
                  {profileError ||
                    "Enter a valid customer ID to load profile data."}
                </div>
              )}
            </div>

            <button
              className="analyze-btn"
              onClick={analyzeTransaction}
              disabled={loading}
              type="button"
            >
              <span>
                {loading
                  ? "Running AI Pipeline..."
                  : "Analyze Transaction"}
              </span>

              <span className="button-arrow">
                →
              </span>
            </button>

            {error && (
              <div className="error-box">
                <strong>Analysis failed</strong>
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="card result-card">
            <div className="card-heading">
              <div>
                <span className="step">02</span>
                <h3>AI Decision</h3>
              </div>

              <span className="small-label">
                LIVE
              </span>
            </div>

            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">
                  ✦
                </div>

                <h4>
                  Decision engine ready
                </h4>

                <p>
                  Analyze a failed payment to run
                  customer enrichment, Risk, Recovery,
                  policy checks, and Supabase audit
                  persistence.
                </p>

                <div className="pipeline-mini">
                  <span>Customer</span>
                  <b>→</b>
                  <span>ML</span>
                  <b>→</b>
                  <span>Policy</span>
                  <b>→</b>
                  <span>Decision</span>
                </div>
              </div>
            ) : (
              <>
                <div
                  className={`decision-banner ${decisionClass}`}
                >
                  <div className="decision-icon">
                    {result.action === "RECOVER"
                      ? "↗"
                      : result.action === "BLOCK"
                        ? "!"
                        : result.action === "REVIEW"
                          ? "?"
                          : "✓"}
                  </div>

                  <div className="decision-content">
                    <span>
                      RECOMMENDED ACTION
                    </span>

                    <strong>
                      {result.action}
                    </strong>

                    <small>
                      {result.priority
                        ? `${result.priority.toUpperCase()} PRIORITY`
                        : "POLICY DECISION"}
                    </small>
                  </div>
                </div>

                <div className="timeline-box">
                  <div className="timeline-header">
                    <div>
                      <span>DECISION PIPELINE</span>

                      <strong>
                        End-to-end execution trace
                      </strong>
                    </div>

                    <span className="timeline-live">
                      LIVE
                    </span>
                  </div>

                  <div className="timeline">
                    {getTimelineSteps(
                      result,
                      customerProfile
                    ).map((step, index, steps) => (
                      <div
                        className={`timeline-step ${step.status}`}
                        key={step.label}
                      >
                        <div className="timeline-node">
                          {step.status === "complete"
                            ? "✓"
                            : step.status === "warning"
                              ? "!"
                              : step.status === "skipped"
                                ? "–"
                                : "·"}
                        </div>

                        <div className="timeline-content">
                          <strong>{step.label}</strong>
                          <span>{step.detail}</span>
                        </div>

                        {index < steps.length - 1 && (
                          <div className="timeline-connector"></div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="score-grid">
                  <div className="score-box">
                    <div className="score-top">
                      <span>Risk Score</span>

                      <strong>
                        {riskPercent}%
                      </strong>
                    </div>

                    <div className="meter">
                      <div
                        style={{
                          width: `${riskWidth}%`,
                        }}
                      ></div>
                    </div>

                    <small>
                      Fraud / risk probability
                    </small>
                  </div>

                  <div className="score-box">
                    <div className="score-top">
                      <span>
                        Recovery Probability
                      </span>

                      <strong>
                        {recoveryPercent}%
                      </strong>
                    </div>

                    <div className="meter recovery">
                      <div
                        style={{
                          width: `${recoveryWidth}%`,
                        }}
                      ></div>
                    </div>

                    <small>
                      Likelihood of successful recovery
                    </small>
                  </div>
                </div>

                <div className="reason-box">
                  <div className="box-label">
                    WHY THIS DECISION
                  </div>

                  <p>
                    {decisionExplanation}
                  </p>

                  <div className="reason-meta">
                    <div>
                      <span>MODEL REASON</span>

                      <code>
                        {result.reason_code}
                      </code>
                    </div>

                    <div>
                      <span>PRIORITY</span>

                      <strong>
                        {result.priority?.toUpperCase() ||
                          "—"}
                      </strong>
                    </div>
                  </div>
                </div>

                {policy && (
                  <div className="policy-box">
                    <div className="policy-header">
                      <div>
                        <span>
                          MERCHANT POLICY
                        </span>

                        <strong>
                          {policy.risk_tolerance_label?.toUpperCase()}
                        </strong>
                      </div>

                      <span className="policy-live">
                        ACTIVE
                      </span>
                    </div>

                    <div className="policy-grid">
                      <div>
                        <span>
                          Risk tolerance
                        </span>

                        <strong>
                          {(
                            result.merchant_risk_tolerance *
                            100
                          ).toFixed(0)}
                          %
                        </strong>
                      </div>

                      <div>
                        <span>
                          Recovery threshold
                        </span>

                        <strong>
                          {(
                            policy.min_recovery_probability_for_auto_action *
                            100
                          ).toFixed(0)}
                          %
                        </strong>
                      </div>

                      <div>
                        <span>
                          Max retries
                        </span>

                        <strong>
                          {policy.max_auto_retries}
                        </strong>
                      </div>

                      <div>
                        <span>Auto retry</span>

                        <strong>
                          {policy.auto_retry_enabled
                            ? "ON"
                            : "OFF"}
                        </strong>
                      </div>
                    </div>
                  </div>
                )}

                {persistence && (
                  <div className="audit-box">
                    <div className="audit-header">
                      <div>
                        <span>
                          AUDIT & PERSISTENCE
                        </span>

                        <small>
                          Transaction flow saved to
                          Supabase
                        </small>
                      </div>

                      <span className="audit-status">
                        ● SAVED
                      </span>
                    </div>

                    <div className="audit-list">
                      {[
                        [
                          "Transaction",
                          persistence.transaction_id,
                          "transaction",
                        ],
                        [
                          "Payment Attempt",
                          persistence.payment_attempt_id,
                          "attempt",
                        ],
                        [
                          "Decision",
                          persistence.decision_id,
                          "decision",
                        ],
                        [
                          "Audit Event",
                          persistence.audit_event_id,
                          "audit",
                        ],
                      ].map(
                        ([label, value, key]) => (
                          <div
                            className="audit-row"
                            key={key}
                          >
                            <span>{label}</span>

                            <code title={value}>
                              {value}
                            </code>

                            <button
                              type="button"
                              className="copy-btn"
                              onClick={() =>
                                copyValue(
                                  value,
                                  key
                                )
                              }
                            >
                              {copied === key
                                ? "Copied"
                                : "Copy"}
                            </button>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </section>

        <footer>
          <span>RevenueRescue AI</span>

          <span>
            ML inference · Decision engine · Supabase audit
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;