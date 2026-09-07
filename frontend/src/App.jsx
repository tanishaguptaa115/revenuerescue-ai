import { useState } from "react";
import "./App.css";

const MERCHANT_ID = "985d35c9-fdbb-4d2a-b974-a71c72b86fae";

const initialForm = {
  amount: 1061.65,
  paymentMethod: "UPI",
  failureReason: "insufficient_funds",
  retryCount: 1,
  isSoftFailure: true,

  numPaymentMethodsUsedRecently: 1,
  ipCountryMismatch: false,
  deviceChangeFlag: false,
  isNewCustomer: false,
  velocityTxnCount1h: 1,
  velocityTxnCount24h: 2,
  daysSinceLastSuccessfulPayment: 5,
  hasPriorSuccess: 1,
  chargebackHistoryCount: 0,

  customerPastSuccessRate: 0.82,
  customerPastRecoveryRate: 0.55,
  nudgeIgnoreTendency: 0.18,
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

    const customerAverage = Math.max(form.amount, 1000);
    const ratio = Number(form.amount) / customerAverage;

    const payload = {
      merchant_id: MERCHANT_ID,
      amount: Number(form.amount),
      currency: "INR",
      payment_method: form.paymentMethod,
      payment_failed: true,
      failure_reason: form.failureReason,
      is_soft_failure: form.isSoftFailure,
      retry_count_so_far: Number(form.retryCount),

      risk_features: {
        amount: Number(form.amount),
        amount_to_customer_avg_ratio: ratio,
        payment_method: form.paymentMethod,
        num_payment_methods_used_recently:
          Number(form.numPaymentMethodsUsedRecently),
        ip_country_mismatch: form.ipCountryMismatch,
        device_change_flag: form.deviceChangeFlag,
        is_new_customer: form.isNewCustomer,
        velocity_txn_count_1h: Number(form.velocityTxnCount1h),
        velocity_txn_count_24h: Number(form.velocityTxnCount24h),
        days_since_last_successful_payment:
          Number(form.daysSinceLastSuccessfulPayment),
        has_prior_success: Number(form.hasPriorSuccess),
        chargeback_history_count: Number(form.chargebackHistoryCount),
      },

      recovery_features: {
        amount: Number(form.amount),
        amount_to_customer_avg_ratio: ratio,
        payment_method: form.paymentMethod,
        failure_reason_code: form.failureReason,
        is_soft_failure: form.isSoftFailure,
        retry_count_so_far: Number(form.retryCount),
        days_since_last_successful_payment:
          Number(form.daysSinceLastSuccessfulPayment),
        has_prior_success: Number(form.hasPriorSuccess),
        is_new_customer: form.isNewCustomer,
        customer_past_success_rate: Number(form.customerPastSuccessRate),
        customer_past_recovery_rate: Number(form.customerPastRecoveryRate),
        nudge_ignore_tendency: Number(form.nudgeIgnoreTendency),
      },
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/score-and-decide", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
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

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand">
            <span className="brand-mark">R</span>
            <div>
              <h1>RevenueRescue AI</h1>
              <p>Autonomous payment recovery & risk intelligence</p>
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
          <span className="eyebrow">MERCHANT CONTROL CENTER</span>
          <h2>Stop revenue leakage before it becomes lost revenue.</h2>
          <p>
            Analyze a failed payment with the trained Risk and Recovery
            models, apply merchant policy, and generate an auditable action.
          </p>
        </section>

        <section className="main-grid">
          <div className="card input-card">
            <div className="card-heading">
              <div>
                <span className="step">01</span>
                <h3>Transaction Signals</h3>
              </div>
              <span className="small-label">INPUT</span>
            </div>

            <div className="form-grid">
              <label>
                Amount
                <div className="input-wrap">
                  <span>₹</span>
                  <input
                    type="number"
                    value={form.amount}
                    onChange={(e) =>
                      update("amount", e.target.value)
                    }
                  />
                </div>
              </label>

              <label>
                Payment Method
                <select
                  value={form.paymentMethod}
                  onChange={(e) =>
                    update("paymentMethod", e.target.value)
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
                    update("failureReason", e.target.value)
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

              <label>
                Retry Count
                <input
                  type="number"
                  min="0"
                  value={form.retryCount}
                  onChange={(e) =>
                    update("retryCount", e.target.value)
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
                    update("velocityTxnCount1h", e.target.value)
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
                    update("velocityTxnCount24h", e.target.value)
                  }
                />
              </label>

              <label>
                Recent Payment Methods
                <input
                  type="number"
                  min="0"
                  value={form.numPaymentMethodsUsedRecently}
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
                  value={form.daysSinceLastSuccessfulPayment}
                  onChange={(e) =>
                    update(
                      "daysSinceLastSuccessfulPayment",
                      e.target.value
                    )
                  }
                />
              </label>
            </div>

            <div className="toggle-row">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={form.isSoftFailure}
                  onChange={(e) =>
                    update("isSoftFailure", e.target.checked)
                  }
                />
                <span></span>
                Soft Failure
              </label>

              <label className="toggle">
                <input
                  type="checkbox"
                  checked={form.isNewCustomer}
                  onChange={(e) =>
                    update("isNewCustomer", e.target.checked)
                  }
                />
                <span></span>
                New Customer
              </label>

              <label className="toggle">
                <input
                  type="checkbox"
                  checked={form.ipCountryMismatch}
                  onChange={(e) =>
                    update("ipCountryMismatch", e.target.checked)
                  }
                />
                <span></span>
                Country Mismatch
              </label>

              <label className="toggle">
                <input
                  type="checkbox"
                  checked={form.deviceChangeFlag}
                  onChange={(e) =>
                    update("deviceChangeFlag", e.target.checked)
                  }
                />
                <span></span>
                Device Changed
              </label>
            </div>

            <div className="advanced-grid">
              <label>
                Customer Success Rate
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={form.customerPastSuccessRate}
                  onChange={(e) =>
                    update(
                      "customerPastSuccessRate",
                      e.target.value
                    )
                  }
                />
              </label>

              <label>
                Customer Recovery Rate
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={form.customerPastRecoveryRate}
                  onChange={(e) =>
                    update(
                      "customerPastRecoveryRate",
                      e.target.value
                    )
                  }
                />
              </label>

              <label>
                Nudge Ignore Tendency
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={form.nudgeIgnoreTendency}
                  onChange={(e) =>
                    update("nudgeIgnoreTendency", e.target.value)
                  }
                />
              </label>

              <label>
                Chargeback History
                <input
                  type="number"
                  min="0"
                  value={form.chargebackHistoryCount}
                  onChange={(e) =>
                    update(
                      "chargebackHistoryCount",
                      e.target.value
                    )
                  }
                />
              </label>
            </div>

            <button
              className="analyze-btn"
              onClick={analyzeTransaction}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Transaction"}
              <span>→</span>
            </button>

            {error && <div className="error-box">{error}</div>}
          </div>

          <div className="card result-card">
            <div className="card-heading">
              <div>
                <span className="step">02</span>
                <h3>AI Decision</h3>
              </div>
              <span className="small-label">LIVE</span>
            </div>

            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">✦</div>
                <h4>Ready for analysis</h4>
                <p>
                  Submit a transaction to run the trained Risk,
                  Recovery, and Decision Engine pipeline.
                </p>
              </div>
            ) : (
              <>
                <div className={`decision-banner ${decisionClass}`}>
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
                    <span>RECOMMENDED ACTION</span>
                    <strong>{result.action}</strong>
                  </div>
                </div>

                <div className="score-grid">
                  <div className="score-box">
                    <span>Risk Score</span>
                    <strong>
                      {(result.risk_score * 100).toFixed(1)}%
                    </strong>
                    <div className="meter">
                      <div
                        style={{
                          width: `${result.risk_score * 100}%`,
                        }}
                      ></div>
                    </div>
                  </div>

                  <div className="score-box">
                    <span>Recovery Probability</span>
                    <strong>
                      {result.recovery_score == null
                        ? "—"
                        : `${(
                            result.recovery_score * 100
                          ).toFixed(1)}%`}
                    </strong>
                    <div className="meter recovery">
                      <div
                        style={{
                          width: `${
                            (result.recovery_score || 0) * 100
                          }%`,
                        }}
                      ></div>
                    </div>
                  </div>
                </div>

                <div className="reason-box">
                  <span>WHY?</span>
                  <p>{result.human_readable_reason}</p>
                </div>

                <div className="policy-box">
                  <div>
                    <span>Merchant Risk Tolerance</span>
                    <strong>
                      {(
                        result.merchant_risk_tolerance * 100
                      ).toFixed(0)}
                      %
                    </strong>
                  </div>
                  <div>
                    <span>Decision Priority</span>
                    <strong>{result.priority}</strong>
                  </div>
                  <div>
                    <span>Reason Code</span>
                    <strong>{result.reason_code}</strong>
                  </div>
                </div>
              </>
            )}
          </div>
        </section>

        {result && (
          <section className="card audit-card">
            <div className="card-heading">
              <div>
                <span className="step">03</span>
                <h3>Audit Trail</h3>
              </div>
              <span className="small-label">SUPABASE</span>
            </div>

            <div className="audit-grid">
              <div>
                <span>Transaction</span>
                <strong>✓ Persisted</strong>
                <small>
                  {result.metadata?.persistence?.transaction_id}
                </small>
              </div>

              <div>
                <span>Payment Attempt</span>
                <strong>✓ Persisted</strong>
                <small>
                  {result.metadata?.persistence?.payment_attempt_id}
                </small>
              </div>

              <div>
                <span>Agent Decision</span>
                <strong>✓ Persisted</strong>
                <small>
                  {result.metadata?.persistence?.decision_id}
                </small>
              </div>

              <div>
                <span>Audit Event</span>
                <strong>✓ Persisted</strong>
                <small>
                  {result.metadata?.persistence?.audit_event_id}
                </small>
              </div>
            </div>
          </section>
        )}

        <footer>
          <span>RevenueRescue AI</span>
          <span>ML · Policy · Decisioning · Audit</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
