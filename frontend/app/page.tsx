"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  CircleDollarSign,
  Cloud,
  Database,
  Gauge,
  LineChart as LineChartIcon,
  Server,
  ShieldCheck,
  Skull,
  Target,
  TrendingDown,
  Zap,
} from "lucide-react";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";


// API

const API_URL =
  typeof window !== "undefined" && window.location.port === "3000"
    ? `http://${window.location.hostname}:8000/api`
    : "/api";


// TYPES

type ZombieServer = {
  server_id: string;
  cpu_utilization: number;
  memory_usage: number;
  cost_per_hour: number;
  cluster: number;
  is_zombie: boolean;
};

type RegressionModel = {
  Model: string;
  "MAE (% CPU)": number;
  "RMSE (% CPU)": number;
  "R2 Score": number;
};

type DashboardData = {
  total_hours: number;
  total_current_cost: number;
  potential_savings: number;
  optimized_cost: number;
  zombie_servers: ZombieServer[];
  servers: string[];
  regression: RegressionModel[];
};

type ForecastPoint = {
  Time: string;
  "Current Telemetry (%)": number;
  "XGBoost Prediction Envelop (%)": number;
};

type ForecastResponse = {
  server: string;
  forecast: ForecastPoint[];
};


// MAIN PAGE

export default function Home() {
  const [activeTab, setActiveTab] = useState<"command" | "benchmark">(
    "command"
  );

  const [dashboard, setDashboard] = useState<DashboardData | null>(null);

  const [selectedServer, setSelectedServer] = useState("");

  const [forecast, setForecast] = useState<ForecastPoint[]>([]);

  const [loading, setLoading] = useState(true);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [error, setError] = useState("");

// Load dashboard

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(`${API_URL}/dashboard`);

        if (!response.ok) {
          throw new Error("Failed to load dashboard data.");
        }

        const data: DashboardData = await response.json();

        setDashboard(data);

        if (data.servers.length > 0) {
          setSelectedServer(data.servers[0]);
        }
      } catch (err) {
        console.error(err);

        setError(
          "Unable to connect to the FinOpsLens API."
        );
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, []);

// Load XGBoost forecast

  useEffect(() => {
    if (!selectedServer) {
      return;
    }

    const loadForecast = async () => {
      try {
        setForecastLoading(true);

        const response = await fetch(
          `${API_URL}/predict?server=${encodeURIComponent(
            selectedServer
          )}`
        );

        if (!response.ok) {
          throw new Error("Failed to load forecast.");
        }

        const data: ForecastResponse = await response.json();

        setForecast(data.forecast);
      } catch (err) {
        console.error(err);
        setForecast([]);
      } finally {
        setForecastLoading(false);
      }
    };

    loadForecast();
  }, [selectedServer]);

// Calculated values

  const zombieCount = useMemo(() => {
    return (
      dashboard?.zombie_servers.filter(
        (server) => server.is_zombie
      ).length ?? 0
    );
  }, [dashboard]);

  const activeServerCount = dashboard?.servers.length ?? 0;

// Loading

  if (loading) {
    return (
      <main className="min-h-screen bg-[#080c14] text-white flex items-center justify-center">
        <div className="text-center">

          <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />

          <p className="text-sm font-semibold tracking-wide text-slate-300">
            INITIALIZING FINOPSLENS
          </p>

          <p className="mt-2 text-xs text-slate-500">
            Connecting to ML intelligence layer...
          </p>

        </div>
      </main>
    );
  }

// Error

  if (error || !dashboard) {
    return (
      <main className="min-h-screen bg-[#080c14] text-white flex items-center justify-center px-6">

        <div className="max-w-lg rounded-2xl border border-red-900/60 bg-[#101722] p-8 text-center">

          <AlertTriangle
            className="mx-auto mb-4 text-red-400"
            size={42}
          />

          <h1 className="text-xl font-bold">
            FinOpsLens API Connection Failed
          </h1>

          <p className="mt-3 text-sm leading-6 text-slate-400">
            {error || "Dashboard data could not be loaded."}
          </p>

          <button
            onClick={() => window.location.reload()}
            className="mt-5 rounded-lg border border-slate-700 bg-slate-800 px-5 py-2.5 text-sm font-semibold transition hover:bg-slate-700"
          >
            Retry Connection
          </button>

        </div>

      </main>
    );
  }

// Dashboard

  return (
    <main className="min-h-screen bg-[#080c14] text-white">

      {/* HEADER */}

      <header className="border-b border-slate-800 px-8 py-5">

        <div className="mx-auto flex max-w-7xl items-center justify-between">

          <div className="flex items-center gap-3">

            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-500/30 bg-cyan-500/10">
              <Cloud className="text-cyan-400" size={21} />
            </div>

            <div>

              <h1 className="text-xl font-bold tracking-tight">
                FinOps<span className="text-cyan-400">Lens</span>
              </h1>

              <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-slate-500">
                Autonomous Cloud Cost Optimizer
              </p>

            </div>

          </div>


          <div className="flex items-center gap-3">

            <div className="flex items-center gap-2 rounded-full border border-emerald-900/50 bg-emerald-500/5 px-3 py-1.5">

              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />

              <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400">
                ML Engine Online
              </span>

            </div>

            <div className="rounded-lg border border-slate-800 bg-[#101722] px-3 py-1.5 text-[10px] font-bold text-slate-500">
              LIVE
            </div>

          </div>

        </div>

      </header>


      {/* OBJECTIVES */}

      <section className="mx-auto max-w-7xl px-8 pt-8">

        <div className="mb-5 flex items-end justify-between">

          <div>

            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-400">
              SYSTEM OBJECTIVES
            </p>

            <h2 className="mt-1 text-2xl font-bold">
              Cloud Financial Intelligence
            </h2>

          </div>

          <div className="text-right text-xs text-slate-500">
            <div>{activeServerCount} active servers</div>
            <div>{zombieCount} waste candidates detected</div>
          </div>

        </div>


        <div className="grid gap-4 md:grid-cols-4">

          <Objective
            icon={<CircleDollarSign size={18} />}
            title="Cost Automation"
            description="Continuously quantify cloud expenditure."
          />

          <Objective
            icon={<Skull size={18} />}
            title="Waste Elimination"
            description="Identify idle and zombie infrastructure."
          />

          <Objective
            icon={<TrendingDown size={18} />}
            title="Proactive Scaling"
            description="Predict future resource requirements."
          />

          <Objective
            icon={<Database size={18} />}
            title="Data Resolution"
            description="Convert telemetry into financial insight."
          />

        </div>

      </section>


      {/* TABS */}

      <div className="mx-auto mt-8 max-w-7xl px-8">

        <div className="flex gap-2 border-b border-slate-800">

          <button
            onClick={() => setActiveTab("command")}
            className={`border-b-2 px-5 py-3 text-xs font-bold uppercase tracking-widest transition ${
              activeTab === "command"
                ? "border-cyan-400 text-cyan-400"
                : "border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            Executive Command Center
          </button>

          <button
            onClick={() => setActiveTab("benchmark")}
            className={`border-b-2 px-5 py-3 text-xs font-bold uppercase tracking-widest transition ${
              activeTab === "benchmark"
                ? "border-cyan-400 text-cyan-400"
                : "border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            Advanced ML Benchmarking
          </button>

        </div>

      </div>


      {/* COMMAND CENTER */}

      {activeTab === "command" && (
        <CommandCenter
          dashboard={dashboard}
          selectedServer={selectedServer}
          setSelectedServer={setSelectedServer}
          forecast={forecast}
          forecastLoading={forecastLoading}
        />
      )}


      {/* BENCHMARK */}

      {activeTab === "benchmark" && (
        <MLBenchmarking dashboard={dashboard} />
      )}

    </main>
  );
}


// OBJECTIVE

function Objective({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="group rounded-xl border border-slate-800 bg-[#101722] p-5 transition hover:border-cyan-900/70">

      <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-900/60 bg-cyan-500/5 text-cyan-400">
        {icon}
      </div>

      <h3 className="text-sm font-bold">
        {title}
      </h3>

      <p className="mt-2 text-xs leading-5 text-slate-500">
        {description}
      </p>

    </div>
  );
}


// COMMAND CENTER

function CommandCenter({
  dashboard,
  selectedServer,
  setSelectedServer,
  forecast,
  forecastLoading,
}: {
  dashboard: DashboardData;
  selectedServer: string;
  setSelectedServer: (server: string) => void;
  forecast: ForecastPoint[];
  forecastLoading: boolean;
}) {
  const zombieServers = dashboard.zombie_servers;

  const zombieCount = zombieServers.filter(
    (server) => server.is_zombie
  ).length;

  return (
    <section className="mx-auto max-w-7xl px-8 py-7">

      {/* METRICS */}

      <SectionTitle
        icon={<Gauge size={19} />}
        title="Executive Command Center"
        subtitle="REAL-TIME FINANCIAL TELEMETRY"
      />

      <div className="mt-5 grid gap-4 md:grid-cols-3">

        <MetricCard
          icon={<CircleDollarSign size={17} />}
          label="TOTAL SPEND"
          value={`$${dashboard.total_current_cost.toFixed(2)}`}
        />

        <MetricCard
          icon={<AlertTriangle size={17} />}
          label="FINANCIAL WASTE"
          value={`$${dashboard.potential_savings.toFixed(2)}`}
          color="red"
        />

        <MetricCard
          icon={<Target size={17} />}
          label="OPTIMIZED TARGET"
          value={`$${dashboard.optimized_cost.toFixed(2)}`}
          color="green"
        />

      </div>


      {/* ZOMBIE TABLE */}

      <div className="mt-7 rounded-2xl border border-slate-800 bg-[#101722] p-6">

        <div className="mb-5 flex items-center justify-between">

          <SectionTitle
            icon={<Skull size={19} />}
            title="Unsupervised Anomaly Tracker"
            subtitle="DBSCAN INFRASTRUCTURE ANALYSIS"
          />

          <div className="rounded-full border border-red-900/50 bg-red-500/5 px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-red-400">
            {zombieCount} waste candidate
            {zombieCount !== 1 ? "s" : ""}
          </div>

        </div>


        <div className="overflow-x-auto">

          <table className="w-full min-w-[700px] text-left">

            <thead>

              <tr className="border-b border-slate-800 text-[10px] font-bold uppercase tracking-wider text-slate-500">

                <th className="px-4 py-3">
                  Server
                </th>

                <th className="px-4 py-3">
                  CPU
                </th>

                <th className="px-4 py-3">
                  Memory %
                </th>

                <th className="px-4 py-3">
                  Hourly Cost
                </th>

                <th className="px-4 py-3">
                  Cluster
                </th>

                <th className="px-4 py-3">
                  Status
                </th>

              </tr>

            </thead>


            <tbody>

              {zombieServers.map((server) => (
                <ServerRow
                  key={server.server_id}
                  id={server.server_id}
                  cpu={server.cpu_utilization.toFixed(2)}
                  memory={server.memory_usage.toFixed(2)}
                  cost={`$${server.cost_per_hour.toFixed(3)}`}
                  cluster={String(server.cluster)}
                  isZombie={server.is_zombie}
                />
              ))}

            </tbody>

          </table>

        </div>

      </div>


      {/* ALERT */}

      {zombieCount > 0 && (
        <div className="mt-5 flex items-start gap-4 rounded-2xl border border-red-900/50 bg-red-500/5 p-5">

          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-red-900/60 bg-red-500/10">
            <Zap className="text-red-400" size={19} />
          </div>

          <div>

            <p className="text-sm font-bold text-red-300">
              Optimization Opportunity Detected
            </p>

            <p className="mt-1 text-xs leading-5 text-slate-400">

              DBSCAN identified{" "}

              <span className="font-semibold text-white">
                {zombieCount}
              </span>{" "}

              anomalous infrastructure node
              {zombieCount !== 1 ? "s" : ""}.

              Eliminating the detected waste could reduce current
              cloud spend by approximately{" "}

              <span className="font-semibold text-red-300">
                ${dashboard.potential_savings.toFixed(2)}
              </span>
              .

            </p>

          </div>

        </div>
      )}


      {/* FORECASTER */}

      <div className="mt-7 rounded-2xl border border-slate-800 bg-[#101722] p-6">

        <SectionTitle
          icon={<LineChartIcon size={19} />}
          title="Proactive Autoscaling Forecaster"
          subtitle="XGBOOST CPU PREDICTION ENGINE"
        />


        <div className="mt-5">

          <label className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
            Select Core Enterprise Node
          </label>

          <select
            value={selectedServer}
            onChange={(event) =>
              setSelectedServer(event.target.value)
            }
            className="mt-2 w-full rounded-lg border border-slate-700 bg-[#0c131e] px-4 py-3 text-sm text-white outline-none transition focus:border-cyan-500"
          >

            {dashboard.servers.map((server) => (
              <option key={server} value={server}>
                {server}
              </option>
            ))}

          </select>

        </div>


        <div className="mt-5 h-[360px] rounded-xl border border-slate-800 bg-[#0b111b] p-4">

          {forecastLoading ? (

            <div className="flex h-full items-center justify-center">

              <div className="text-center">

                <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />

                <p className="text-xs font-semibold text-slate-400">
                  Loading XGBoost forecast...
                </p>

              </div>

            </div>

          ) : forecast.length === 0 ? (

            <div className="flex h-full items-center justify-center">

              <div className="text-center">

                <Activity
                  className="mx-auto mb-3 text-slate-600"
                  size={30}
                />

                <p className="text-sm font-semibold text-slate-400">
                  No forecast data available
                </p>

              </div>

            </div>

          ) : (

            <ResponsiveContainer
              width="100%"
              height="100%"
            >

              <LineChart
                data={forecast}
                margin={{
                  top: 10,
                  right: 20,
                  left: 0,
                  bottom: 10,
                }}
              >

                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1e293b"
                />

                <XAxis
                  dataKey="Time"
                  tick={{
                    fill: "#64748b",
                    fontSize: 10,
                  }}
                  axisLine={{
                    stroke: "#334155",
                  }}
                  tickLine={false}
                />

                <YAxis
                  domain={[0, "auto"]}
                  tick={{
                    fill: "#64748b",
                    fontSize: 10,
                  }}
                  axisLine={{
                    stroke: "#334155",
                  }}
                  tickLine={false}
                  unit="%"
                />

                <Tooltip
                  contentStyle={{
                    backgroundColor: "#101722",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                    color: "#fff",
                    fontSize: "12px",
                  }}
                />

                <Legend
                  wrapperStyle={{
                    fontSize: "11px",
                    color: "#94a3b8",
                  }}
                />

                <Line
                  type="monotone"
                  dataKey="Current Telemetry (%)"
                  stroke="#94a3b8"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />

                <Line
                  type="monotone"
                  dataKey="XGBoost Prediction Envelop (%)"
                  stroke="#22d3ee"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 5 }}
                />

              </LineChart>

            </ResponsiveContainer>

          )}

        </div>


        <div className="mt-4 flex items-center gap-2 text-[10px] text-slate-500">

          <ShieldCheck
            size={13}
            className="text-cyan-400"
          />

          Forecast generated using the trained XGBoost model.

        </div>

      </div>

    </section>
  );
}


// METRIC CARD

function MetricCard({
  icon,
  label,
  value,
  color = "default",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color?: "default" | "red" | "green";
}) {
  const iconColor =
    color === "red"
      ? "text-red-400"
      : color === "green"
      ? "text-emerald-400"
      : "text-cyan-400";

  return (
    <div className="rounded-2xl border border-slate-800 bg-[#101722] p-6">

      <div className="flex items-center justify-between">

        <div
          className={`flex h-9 w-9 items-center justify-center rounded-lg border border-slate-800 bg-black/10 ${iconColor}`}
        >
          {icon}
        </div>

        <Activity
          size={15}
          className="text-slate-700"
        />

      </div>

      <p className="mt-5 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
        {label}
      </p>

      <p className="mt-2 text-3xl font-bold tracking-tight">
        {value}
      </p>

    </div>
  );
}


// SERVER ROW

function ServerRow({
  id,
  cpu,
  memory,
  cost,
  cluster,
  isZombie,
}: {
  id: string;
  cpu: string;
  memory: string;
  cost: string;
  cluster: string;
  isZombie: boolean;
}) {
  return (
    <tr className="border-b border-slate-800/70 text-sm transition hover:bg-white/[0.02]">

      <td className="px-4 py-4">

        <div className="flex items-center gap-3">

          <div
            className={`flex h-8 w-8 items-center justify-center rounded-lg ${
              isZombie
                ? "bg-red-500/10 text-red-400"
                : "bg-cyan-500/5 text-cyan-400"
            }`}
          >
            <Server size={15} />
          </div>

          <span className="font-medium text-slate-200">
            {id}
          </span>

        </div>

      </td>


      <td className="px-4 py-4 text-slate-400">
        {cpu}%
      </td>


      <td className="px-4 py-4 text-slate-400">
        {memory}%
      </td>


      <td className="px-4 py-4 font-mono text-slate-400">
        {cost}
      </td>


      <td className="px-4 py-4">

        <span className="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-400">
          {cluster}
        </span>

      </td>


      <td className="px-4 py-4">

        {isZombie ? (

          <span className="inline-flex items-center gap-1.5 rounded-full border border-red-900/50 bg-red-500/5 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-red-400">

            <Skull size={11} />

            Zombie

          </span>

        ) : (

          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-900/50 bg-emerald-500/5 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-emerald-400">

            <CheckCircle2 size={11} />

            Active

          </span>

        )}

      </td>

    </tr>
  );
}


// SECTION TITLE

function SectionTitle({
  icon,
  title,
  subtitle,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="flex items-center gap-3">

      <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-800 bg-black/10 text-cyan-400">
        {icon}
      </div>

      <div>

        <h2 className="text-base font-bold">
          {title}
        </h2>

        <p className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.18em] text-slate-600">
          {subtitle}
        </p>

      </div>

    </div>
  );
}


// ML BENCHMARKING

function MLBenchmarking({
  dashboard,
}: {
  dashboard: DashboardData;
}) {
  const xgboost = dashboard.regression.find(
    (model) => model.Model === "XGBoost"
  );

  const noisePoints = dashboard.zombie_servers.filter(
    (server) => server.cluster === -1
  ).length;

  const zombieCount = dashboard.zombie_servers.filter(
    (server) => server.is_zombie
  ).length;

  return (
    <section className="mx-auto max-w-7xl px-8 py-7">

      <SectionTitle
        icon={<BrainCircuit size={19} />}
        title="Advanced ML Benchmarking"
        subtitle="MATHEMATICAL MODEL VALIDATION MATRIX"
      />


      {/* MODEL CARDS */}

      <div className="mt-5 grid gap-4 md:grid-cols-3">

        {dashboard.regression.map((model) => (

          <div
            key={model.Model}
            className={`rounded-2xl border p-6 ${
              model.Model === "XGBoost"
                ? "border-cyan-900/60 bg-cyan-500/[0.03]"
                : "border-slate-800 bg-[#101722]"
            }`}
          >

            <div className="flex items-center justify-between">

              <div className="flex items-center gap-3">

                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-800 bg-black/10">

                  {model.Model === "XGBoost" ? (
                    <BrainCircuit
                      size={17}
                      className="text-cyan-400"
                    />
                  ) : (
                    <BarChart3
                      size={17}
                      className="text-slate-500"
                    />
                  )}

                </div>


                <div>

                  <p className="text-sm font-bold">
                    {model.Model}
                  </p>

                  {model.Model === "XGBoost" && (
                    <p className="text-[9px] font-bold uppercase tracking-widest text-cyan-400">
                      Selected Forecaster
                    </p>
                  )}

                </div>

              </div>

            </div>


            <div className="mt-6 grid grid-cols-3 gap-3">

              <BenchmarkValue
                label="MAE"
                value={model["MAE (% CPU)"].toFixed(2)}
              />

              <BenchmarkValue
                label="RMSE"
                value={model["RMSE (% CPU)"].toFixed(2)}
              />

              <BenchmarkValue
                label="R²"
                value={model["R2 Score"].toFixed(4)}
              />

            </div>

          </div>

        ))}

      </div>


      {/* VALIDATION */}

      <div className="mt-7 grid gap-5 md:grid-cols-2">

        <ProofCard
          icon={<Database size={19} />}
          title="K-Means Metric Limitations"
          text="Traditional centroid-based clustering can struggle to isolate irregular low-utilization infrastructure patterns when the anomaly does not fit a compact spherical cluster."
        />

        <ProofCard
          icon={<BrainCircuit size={19} />}
          title="DBSCAN Mathematical Resolution"
          text="Density-based clustering identifies dense operational environments while treating isolated telemetry points as noise, enabling FinOpsLens to surface anomalous infrastructure."
        />

      </div>


      {/* DBSCAN RESULT */}

      <div className="mt-5 rounded-2xl border border-slate-800 bg-[#101722] p-6">

        <div className="flex items-center gap-3">

          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-red-900/50 bg-red-500/5">

            <Skull
              size={18}
              className="text-red-400"
            />

          </div>


          <div>

            <h3 className="text-sm font-bold">
              DBSCAN Anomaly Detection Result
            </h3>

            <p className="mt-1 text-xs text-slate-500">
              Infrastructure telemetry classification
            </p>

          </div>

        </div>


        <div className="mt-6 grid gap-4 md:grid-cols-3">

          <ResultStat
            label="Dense Environments"
            value="2"
          />

          <ResultStat
            label="Noise Points"
            value={String(noisePoints)}
            danger
          />

          <ResultStat
            label="Detected Zombie Servers"
            value={String(zombieCount)}
            danger
          />

        </div>

      </div>


      {/* MODEL SUMMARY */}

      {xgboost && (

        <div className="mt-5 flex items-start gap-4 rounded-2xl border border-cyan-900/50 bg-cyan-500/[0.03] p-6">

          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan-900/60 bg-cyan-500/5">

            <Target
              size={18}
              className="text-cyan-400"
            />

          </div>


          <div>

            <p className="text-sm font-bold">
              XGBoost Forecasting Model
            </p>

            <p className="mt-2 text-xs leading-5 text-slate-400">

              The trained XGBoost model currently reports an MAE of{" "}

              <span className="font-semibold text-white">
                {xgboost["MAE (% CPU)"].toFixed(2)}%
              </span>

              , RMSE of{" "}

              <span className="font-semibold text-white">
                {xgboost["RMSE (% CPU)"].toFixed(2)}%
              </span>

              , and R² of{" "}

              <span className="font-semibold text-cyan-400">
                {xgboost["R2 Score"].toFixed(4)}
              </span>
              .

            </p>

          </div>

        </div>

      )}

    </section>
  );
}


// BENCHMARK VALUE

function BenchmarkValue({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-black/10 p-3">

      <p className="text-[9px] font-bold uppercase tracking-widest text-slate-600">
        {label}
      </p>

      <p className="mt-1 font-mono text-sm font-bold text-slate-300">
        {value}
      </p>

    </div>
  );
}


// RESULT STAT

function ResultStat({
  label,
  value,
  danger = false,
}: {
  label: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-black/10 p-5">

      <p className="text-[9px] font-bold uppercase tracking-widest text-slate-600">
        {label}
      </p>

      <p
        className={`mt-2 text-2xl font-bold ${
          danger
            ? "text-red-400"
            : "text-cyan-400"
        }`}
      >
        {value}
      </p>

    </div>
  );
}


// PROOF CARD

function ProofCard({
  icon,
  title,
  text,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#101722] p-6">

      <div className="flex items-center gap-3">

        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-800 bg-black/10 text-cyan-400">
          {icon}
        </div>

        <h3 className="text-sm font-bold">
          {title}
        </h3>

      </div>

      <p className="mt-4 text-xs leading-6 text-slate-500">
        {text}
      </p>

    </div>
  );
}