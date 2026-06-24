export interface IPOListItem {
  id: string;
  company_name: string;
  symbol?: string;
  status: 'active' | 'upcoming' | 'listed';
  open_date: string;
  close_date: string;
  listing_date?: string;
  issue_price_min: number;
  issue_price_max: number;
  lot_size: number;
  issue_size_cr: number;
  gmp: number;
  gmp_pct: number;
  subscription_total: number;
  subscription_qib: number;
  subscription_nii: number;
  subscription_rii: number;
  recommendation: 'INVEST' | 'AVOID' | 'NEUTRAL';
  recommendation_score: number;
  expected_listing_gain_pct?: number;
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW';
  sector: string;
  registrar: string;
}

export interface IPOFinancials {
  revenue_cr?: number;
  ebitda_cr?: number;
  pat_cr?: number;
  eps?: number;
  pe_ratio?: number;
  roe?: number;
  debt_equity?: number;
}

export interface IPODetail extends IPOListItem {
  company_description: string;
  financials: IPOFinancials;
  promoter_holding_pre?: number;
  promoter_holding_post?: number;
  industry_pe?: number;
  objects_of_issue: string[];
  pros: string[];
  cons: string[];
  analysis_summary: string;
  lead_managers: string[];
  dimension_scores: Record<string, number>;
}
