export interface PredictionResult {
  churn_probability: number;
  risk_tier: string;
  top_reasons: string[];
  model_used: string;
}

export interface CustomerInput {
  gender: number;
  SeniorCitizen: number;
  Partner: number;
  Dependents: number;
  tenure: number;
  PhoneService: number;
  MultipleLines: number;
  OnlineSecurity: number;
  OnlineBackup: number;
  DeviceProtection: number;
  TechSupport: number;
  StreamingTV: number;
  StreamingMovies: number;
  PaperlessBilling: number;
  MonthlyCharges: number;
  TotalCharges: number;
  "InternetService_Fiber optic": number;
  InternetService_No: number;
  "Contract_One year": number;
  "Contract_Two year": number;
  "PaymentMethod_Credit card (automatic)": number;
  "PaymentMethod_Electronic check": number;
  "PaymentMethod_Mailed check": number;
}

export interface ModelResults {
  best_model: string;
  results: Record<string, ModelMetrics>;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface DataOverview {
  rows: number;
  columns: number;
  churn_rate: number;
  features: string[];
}

export interface ChurnRateData {
  group: string;
  churn_rate: number;
  count: number;
}

export interface BusinessImpactResult {
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  true_negatives: number;
  revenue_saved: number;
  retention_spend: number;
  missed_revenue: number;
  net_benefit: number;
  roi_percent: number;
}

export interface ExperimentConfig {
  name: string;
  intervention_type: string;
  intervention_description: string;
  expected_effect_size: number;
  cost_per_customer: number;
  risk_tiers: string[];
  split_ratio: number;
  significance_level: number;
  power: number;
  random_seed: number;
  avg_monthly_revenue: number;
  months_saved: number;
}

export interface ExperimentSummary {
  experiment_id: string;
  name: string;
  status: string;
  intervention_type?: string;
  risk_tiers?: string[];
  sample_size?: number;
  absolute_lift?: number;
  p_value?: number;
  is_significant?: boolean;
  roi_percent?: number;
}

export interface PowerAnalysisResult {
  required_sample_size_per_group: number;
  total_required: number;
  achieved_power: number;
  baseline_churn_rate: number;
  treatment_churn_rate: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  toolResults?: ToolResult[];
}

export interface ToolResult {
  tool: string;
  result: Record<string, unknown>;
}

export interface HighRiskCustomer {
  customer_id: number;
  churn_probability: number;
  risk_tier: string;
  top_reason: string;
  monthly_charges: number;
  tenure: number;
  contract: string;
}

export interface PredictionHistoryItem {
  id: string;
  timestamp: string;
  churn_probability: number;
  risk_tier: string;
  top_reasons: string[];
  model_used: string;
  inputs: Record<string, unknown>;
}
