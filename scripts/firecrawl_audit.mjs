import { Firecrawl } from 'firecrawl';
import { z } from 'zod';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logsDir = path.join(__dirname, '..', 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

const apiKey = process.env.FIRECRAWL_API_KEY;
if (!apiKey) {
  console.error('❌ Error: FIRECRAWL_API_KEY environment variable is not set.');
  process.exit(1);
}

const firecrawl = new Firecrawl({ apiKey });

// Shortened prompt to comply with the 10,000-character API limit
const prompt = `
Perform a scientific and statistical audit of the Zindian Orchestrator's Source of Truth (SoT v2.3) principles and assumptions.
Evaluate whether the orchestrator's gate checks, metric normalizations, spatial handling, and ensembling criteria are grounded in peer-reviewed statistical/ML literature.

Methodology:
- Audit each claim using the Three-Lens Philosophy (General, Specific, Generalisation).
- Restrict sources to peer-reviewed papers (NeurIPS, ICML, ICLR, JMLR) and standard textbooks (Hastie/Tibshirani, Murphy, Bishop).
- Exclude blog posts and forum answers.

Claims to audit:
1. Bessel's correction (ddof=1) for fold score variance: np.var(fold_scores, ddof=1).
- Questions: 1. Does ddof=1 hold for overlapping, non-independent CV splits? 2. What is the correct estimator per Nadeau & Bengio? 3. Finite vs infinite population effect.
- Search: Nadeau Bengio variance cross validation, unbiased estimator k-fold variance.

2. MAPE Zero-Target exclusion: MAPE computed exclusively where y_true != 0.
- Questions: 1. Does removing zero values distort the distribution or bias model selection? 2. What does metrics literature recommend for zero targets (e.g. sMAPE, MASE)?
- Search: symmetric MAPE vs MAPE zero targets, mean absolute scaled error Hyndman.

3. Normalizing RMSE by target standard deviation: normalized_distance = RMSE / target_std.
- Questions: 1. Is RMSE/target_std a valid coefficient of variation? 2. Does combining unbounded RMSE/target_std with bounded 1-F1 violate geometric consistency? How do multi-task frameworks handle this (e.g., Kendall 2018 uncertainty weighting)?
- Search: Kendall multi-task loss uncertainty weighting, relative squared error regression metric.

4. Correlation metrics for candidate diversity: Prune candidates with prediction correlation > 0.95 (Pearson for classification, Spearman for regression).
- Questions: 1. Does ensembling benefit from low prediction correlation or error correlation? 2. Is a 0.95 threshold supported by ensemble pruning theory (Caruana, Kuncheva)?
- Search: ensemble diversity measures Kuncheva, diversity on residuals ensemble pruning.

5. Pseudo-label recombination policies: freeze unaugmented targets or block composite evaluation.
- Questions: 1. Does augmenting only classification targets while freezing regression targets break covariance structures? 2. How is multi-task joint probability consistency handled in self-training?
- Search: semi-supervised multi-task learning target correlation, joint pseudo-labeling.

6. SHAP dominance ratio > 3.0 for leakage detection: mean(|SHAP|) of top feature / mean(|SHAP|) of remaining features > 3.0.
- Questions: 1. Does multicollinearity split SHAP values and hide leaks? 2. Are there better supported leak detection methods in literature?
- Search: SHAP data leakage detection, multicollinearity SHAP feature importance.

7. Spatial signals treated as group signals in GroupKFold.
- Questions: 1. Does GroupKFold on block IDs suffer from spatial autocorrelation bias? 2. What does spatial CV literature (Roberts 2017) recommend (e.g. spatial buffering)?
- Search: spatial cross-validation spatial autocorrelation Roberts, block cross-validation buffer.

8. Top 10% confidence threshold for pseudo-labeling.
- Questions: 1. Does a flat 10% threshold introduce bias under class imbalance? 2. What does semi-supervised tabular literature (Lee 2013, FixMatch) suggest?
- Search: pseudo-labeling confidence threshold tabular data, self-training class imbalance.

9. Absolute gate margin (gate_margin = 0.001) for branch promotion.
- Questions: 1. Is 0.001 statistically distinguishable from noise for standard sample sizes? 2. Should it be defined in standard error units (delta >= 1 SE) instead?
- Search: standard error of cross-validation k-fold, minimum detectable effect classifier.

10. File hashing (SHA-256) for dataset reproducibility verification.
- Questions: 1. Can byte-level hashing fail to catch float representation or parsing differences across OS/CPU architectures? 2. Should schemas and statistical properties be checked instead?
- Search: machine learning dataset reproducibility file hash, floating point representation data drift.
`;

const schema = z.object({
  research_brief_metadata: z.object({
    version: z.string().optional(),
    test_baseline: z.object({
      total_tests: z.number().optional(),
      passed: z.number().optional(),
      skipped: z.number().optional()
    }).optional()
  }).optional(),
  claims: z.array(z.object({
    claim_id: z.number(),
    title: z.string(),
    lens: z.string(),
    sot_mapping: z.string().optional(),
    policy_definition: z.string(),
    justification: z.string().optional(),
    audit_questions: z.array(z.object({
      value: z.string()
    })),
    targeted_search_terms: z.array(z.object({
      value: z.string()
    })).optional()
  })).describe("List of claims to be audited."),
  methodology_constraints: z.object({
    allowed_sources: z.array(z.object({
      value: z.string()
    })).optional(),
    excluded_sources: z.array(z.object({
      value: z.string()
    })).optional()
  })
});

async function runAudit() {
  console.log('🚀 Starting SoT Scientific and Statistical Audit via Firecrawl API...');
  try {
    const result = await firecrawl.agent({
      prompt,
      schema,
      model: 'spark-1-mini',
    });

    const outputPath = path.join(logsDir, 'sot_scientific_audit_result.json');
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf-8');

    console.log(`\n✅ Audit Completed Successfully!`);
    console.log(`Saved structured data to: ${outputPath}`);

    if (result && result.claims) {
      let mdReport = `# SoT v2.3 Scientific and Statistical Audit Report\n\n`;
      mdReport += `**Audit Date:** ${new Date().toLocaleDateString()}\n`;
      mdReport += `**Baseline Tests:** 252 passed, 6 skipped\n\n`;
      mdReport += `## Claims Audit Summary\n\n`;

      for (const claim of result.claims) {
        mdReport += `### CLAIM ${claim.claim_id}: ${claim.title}\n`;
        mdReport += `- **Lens:** ${claim.lens}\n`;
        mdReport += `- **SoT Mapping:** ${claim.sot_mapping || 'N/A'}\n`;
        mdReport += `- **Policy:** \`${claim.policy_definition}\`\n`;
        mdReport += `- **Justification:** ${claim.justification || 'N/A'}\n\n`;

        mdReport += `#### Audit Questions Evaluated:\n`;
        claim.audit_questions.forEach(q => {
          mdReport += `- ${q.value}\n`;
        });
        mdReport += `\n---\n\n`;
      }

      const mdPath = path.join(logsDir, 'sot_scientific_audit_summary.md');
      fs.writeFileSync(mdPath, mdReport, 'utf-8');
      console.log(`Saved summary markdown to: ${mdPath}`);
    }
  } catch (error) {
    console.error('❌ Error during Firecrawl API call:', error);
    process.exit(1);
  }
}

runAudit();
