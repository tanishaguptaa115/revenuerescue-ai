import { useState } from "react";
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

function App() {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const update = (key, value) => {
    setForm((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const analyzeTransaction = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    const daysSinceSuccess = Number(
      form.daysSinceLastSuccessfulPayment
    );

    const hasPriorSuccess =
      Number.isFinite(daysSinceSuccess) && daysSinceSuccess >= 0;

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

  const decisionClass =
    result?.action?.toLowerCase() || "";

  const riskPercent = result
    ? (result.risk_score * 100).toFixed(1)
    : "0.0";

  const recoveryPercent = result
    ? (result.recovery_score * 100).toFixed(1)
    : "0.0";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
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
        </div>

        <div className="status-pill">
          <span className="status-dot"></span>
          AI Engine Online
        </div>
      </header>

      <main className="dashboard">
        <section className="hero-copy">
          <span className="eyebrow">
            MERCHANT CONTROL CENTER
          </span>

          <h2>
            Stop revenue leakage before it becomes lost
            revenue.
          </h2>

          <p>
            Analyze a failed payment with the trained Risk
            and Recovery models, apply merchant policy, and
            generate an auditable action.
          </p>
        </section>

        <section className="main-grid">
          {/* -------------------------------------------------
              INPUT CARD
          -------------------------------------------------- */}
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
              {/* Customer */}
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
                  placeholder="e.g. cust_000000"
                />
              </label>

              {/* Amount */}
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

              {/* Payment method */}
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

              {/* Failure reason */}
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
                </select>
              </label>

              {/* Retry count */}
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

              {/* Velocity 1h */}
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

              {/* Velocity 24h */}
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

              {/* Recent payment methods */}
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

              {/* Days since success */}
              <label>
                Days Since Last Success

                <input
                  type="number"
                  min="0"
                  value={
                    form.daysSinceLastSuccessfulPayment
                  }
                  onChange={(e) =>
                    update(
                      "daysSinceLastSuccessfulPayment",
                      e.target.value
                    )
                  }
                />
              </label>
            </div>

            {/* Toggle signals */}
            <div className="toggle-grid">
              <label className="toggle-card">
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
                    Eligible for recovery attempt
                  </small>
                </span>
              </label>

              <label className="toggle-card">
                <input
                  type="checkbox"
                  checked={form.ipCountryMismatch}
                  onChange={(e) =>
                    update(
                      "ipCountryMismatch",
                      e.target.checked
                    )
                  }
                />

                <span>
                  <strong>Country Mismatch</strong>
                  <small>
                    IP country differs from profile
                  </small>
                </span>
              </label>

              <label className="toggle-card">
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
                    Recent device change signal
                  </small>
                </span>
              </label>
            </div>

            <div className="customer-note">
              <strong>Customer intelligence:</strong>{" "}
              historical customer behavior is loaded
              automatically by the backend from the
              customer profile associated with this ID.
            </div>

            <button
              className="analyze-btn"
              onClick={analyzeTransaction}
              disabled={loading}
            >
              {loading
                ? "Analyzing..."
                : "Analyze Transaction"}

              <span>→</span>
            </button>

            {error && (
              <div className="error-box">
                {error}
              </div>
            )}
          </div>

          {/* -------------------------------------------------
              RESULT CARD
          -------------------------------------------------- */}
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
                <div className="empty-icon">✦</div>

                <h4>Ready for analysis</h4>

                <p>
                  Submit a transaction to run the
                  trained Risk, Recovery, and Decision
                  Engine pipeline.
                </p>
              </div>
            ) : (
              <>
                {/* Decision */}
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

                  <div>
                    <span>
                      RECOMMENDED ACTION
                    </span>

                    <strong>
                      {result.action}
                    </strong>
                  </div>
                </div>

                {/* Model scores */}
                <div className="metric-grid">
                  <div className="metric-card">
                    <span>Risk Score</span>
                    <strong>
                      {riskPercent}%
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>Recovery Probability</span>
                    <strong>
                      {recoveryPercent}%
                    </strong>
                  </div>
                </div>

                {/* Reason */}
                <div className="reason-box">
                  <span>WHY THIS DECISION</span>

                  <strong>
                    {result.human_readable_reason}
                  </strong>

                  <small>
                    Reason code:{" "}
                    {result.reason_code}
                  </small>
                </div>

                {/* Policy */}
                {result.metadata?.merchant_policy && (
                  <div className="policy-box">
                    <div className="policy-header">
                      <span>
                        MERCHANT POLICY
                      </span>

                      <strong>
                        {result.metadata
                          .merchant_policy
                          .risk_tolerance_label
                          ?.toUpperCase()}
                      </strong>
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
                            result.metadata
                              .recovery_score_threshold *
                            100
                          ).toFixed(0)}
                          %
                        </strong>
                      </div>

                      <div>
                        <span>
                          Max auto retries
                        </span>

                        <strong>
                          {
                            result.metadata
                              .merchant_policy
                              .max_auto_retries
                          }
                        </strong>
                      </div>
                    </div>
                  </div>
                )}

                {/* Persistence */}
                {result.metadata?.persistence && (
                  <div className="audit-box">
                    <div className="audit-header">
                      <span>
                        AUDIT & PERSISTENCE
                      </span>

                      <span className="audit-status">
                        ● SAVED
                      </span>
                    </div>

                    <div className="audit-row">
                      <span>Transaction</span>

                      <code>
                        {
                          result.metadata.persistence
                            .transaction_id
                        }
                      </code>
                    </div>

                    <div className="audit-row">
                      <span>Payment Attempt</span>

                      <code>
                        {
                          result.metadata.persistence
                            .payment_attempt_id
                        }
                      </code>
                    </div>

                    <div className="audit-row">
                      <span>Decision</span>

                      <code>
                        {
                          result.metadata.persistence
                            .decision_id
                        }
                      </code>
                    </div>

                    <div className="audit-row">
                      <span>Audit Event</span>

                      <code>
                        {
                          result.metadata.persistence
                            .audit_event_id
                        }
                      </code>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;