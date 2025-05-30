# evaluation/advanced_statistics.py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import matplotlib.pyplot as plt


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for LLAMAREC evaluation"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}

    def comprehensive_comparison(self,
                                 system_results: Dict[str, List[float]],
                                 metric_name: str = "quality") -> Dict:
        """
        Comprehensive statistical comparison between systems

        Args:
            system_results: {system_name: [scores]}
            metric_name: Name of the metric being compared

        Returns:
            Comprehensive statistical analysis results
        """

        analysis = {
            "metric": metric_name,
            "systems": list(system_results.keys()),
            "descriptive_stats": {},
            "normality_tests": {},
            "homoscedasticity_test": {},
            "parametric_tests": {},
            "non_parametric_tests": {},
            "effect_sizes": {},
            "post_hoc_tests": {},
            "recommendations": []
        }

        # 1. Descriptive Statistics
        for system, scores in system_results.items():
            analysis["descriptive_stats"][system] = {
                "n": len(scores),
                "mean": np.mean(scores),
                "std": np.std(scores, ddof=1),
                "median": np.median(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75),
                "min": np.min(scores),
                "max": np.max(scores),
                "skewness": stats.skew(scores),
                "kurtosis": stats.kurtosis(scores)
            }

        # 2. Normality Tests
        for system, scores in system_results.items():
            if len(scores) >= 8:  # Minimum for Shapiro-Wilk
                stat, p_value = stats.shapiro(scores)
                analysis["normality_tests"][system] = {
                    "test": "Shapiro-Wilk",
                    "statistic": stat,
                    "p_value": p_value,
                    "is_normal": p_value > self.alpha
                }

        # 3. Homoscedasticity Test (Equal Variances)
        if len(system_results) >= 2:
            groups = list(system_results.values())
            if all(len(g) >= 3 for g in groups):
                stat, p_value = stats.levene(*groups)
                analysis["homoscedasticity_test"] = {
                    "test": "Levene",
                    "statistic": stat,
                    "p_value": p_value,
                    "equal_variances": p_value > self.alpha
                }

        # 4. Choose appropriate tests based on assumptions
        all_normal = all(
            test.get("is_normal", False)
            for test in analysis["normality_tests"].values()
        )
        equal_variances = analysis.get("homoscedasticity_test", {}).get("equal_variances", False)

        # 5. Parametric Tests (if assumptions met)
        if all_normal and equal_variances and len(system_results) >= 2:
            if len(system_results) == 2:
                # Independent t-test
                group1, group2 = list(system_results.values())
                stat, p_value = stats.ttest_ind(group1, group2)
                analysis["parametric_tests"]["t_test"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant": p_value < self.alpha
                }
            else:
                # One-way ANOVA
                groups = list(system_results.values())
                stat, p_value = stats.f_oneway(*groups)
                analysis["parametric_tests"]["anova"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant": p_value < self.alpha
                }

        # 6. Non-parametric Tests (robust alternatives)
        if len(system_results) == 2:
            group1, group2 = list(system_results.values())
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            analysis["non_parametric_tests"]["mann_whitney"] = {
                "statistic": stat,
                "p_value": p_value,
                "significant": p_value < self.alpha
            }
        elif len(system_results) > 2:
            groups = list(system_results.values())
            stat, p_value = kruskal(*groups)
            analysis["non_parametric_tests"]["kruskal_wallis"] = {
                "statistic": stat,
                "p_value": p_value,
                "significant": p_value < self.alpha
            }

        # 7. Effect Size Calculations
        system_names = list(system_results.keys())
        for i, system1 in enumerate(system_names):
            for system2 in system_names[i + 1:]:
                group1 = system_results[system1]
                group2 = system_results[system2]

                # Cohen's d
                cohens_d = self._calculate_cohens_d(group1, group2)

                # Glass's Delta (when variances are unequal)
                glass_delta = self._calculate_glass_delta(group1, group2)

                # Cliff's Delta (non-parametric effect size)
                cliffs_delta = self._calculate_cliffs_delta(group1, group2)

                analysis["effect_sizes"][f"{system1}_vs_{system2}"] = {
                    "cohens_d": cohens_d,
                    "cohens_d_interpretation": self._interpret_cohens_d(cohens_d),
                    "glass_delta": glass_delta,
                    "cliffs_delta": cliffs_delta,
                    "cliffs_delta_interpretation": self._interpret_cliffs_delta(cliffs_delta)
                }

        # 8. Post-hoc tests (if overall test was significant)
        overall_significant = (
                analysis.get("parametric_tests", {}).get("anova", {}).get("significant", False) or
                analysis.get("non_parametric_tests", {}).get("kruskal_wallis", {}).get("significant", False)
        )

        if overall_significant and len(system_results) > 2:
            analysis["post_hoc_tests"] = self._post_hoc_analysis(system_results)

        # 9. Practical Recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +
                              (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))

        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    def _calculate_glass_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Glass's Delta (uses control group std)"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std2 = np.std(group2, ddof=1)
        return (mean1 - mean2) / std2 if std2 > 0 else 0

    def _calculate_cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's Delta (non-parametric effect size)"""
        n1, n2 = len(group1), len(group2)

        # Count dominance matrix
        dominance = 0
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1

        return dominance / (n1 * n2)

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's Delta effect size"""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"

    def _post_hoc_analysis(self, system_results: Dict[str, List[float]]) -> Dict:
        """Perform post-hoc pairwise comparisons with correction"""
        systems = list(system_results.keys())
        pairwise_results = {}
        p_values = []

        # Pairwise comparisons
        for i, system1 in enumerate(systems):
            for system2 in systems[i + 1:]:
                group1 = system_results[system1]
                group2 = system_results[system2]

                # Use Mann-Whitney U (robust)
                stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

                pairwise_results[f"{system1}_vs_{system2}"] = {
                    "statistic": stat,
                    "p_value_raw": p_value
                }
                p_values.append(p_value)

        # Multiple comparison correction
        if p_values:
            # Benjamini-Hochberg FDR correction
            _, p_corrected_bh, _, _ = multipletests(p_values, method='fdr_bh')

            # Bonferroni correction
            _, p_corrected_bonf, _, _ = multipletests(p_values, method='bonferroni')

            # Add corrected p-values
            for i, (comparison, result) in enumerate(pairwise_results.items()):
                result["p_value_bh_corrected"] = p_corrected_bh[i]
                result["p_value_bonferroni_corrected"] = p_corrected_bonf[i]
                result["significant_bh"] = p_corrected_bh[i] < self.alpha
                result["significant_bonferroni"] = p_corrected_bonf[i] < self.alpha

        return pairwise_results

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate practical recommendations based on analysis"""
        recommendations = []

        # Sample size recommendations
        min_n = min(stats["n"] for stats in analysis["descriptive_stats"].values())
        if min_n < 30:
            recommendations.append(
                f"⚠️ Small sample size detected (min n={min_n}). "
                "Consider collecting more data for robust conclusions."
            )

        # Normality violations
        non_normal = [
            system for system, test in analysis["normality_tests"].items()
            if not test.get("is_normal", True)
        ]
        if non_normal:
            recommendations.append(
                f"📊 Non-normal distributions detected for {non_normal}. "
                "Non-parametric tests are more appropriate."
            )

        # Effect size interpretation
        large_effects = []
        for comparison, effects in analysis.get("effect_sizes", {}).items():
            if effects.get("cohens_d_interpretation") == "large":
                large_effects.append(comparison)

        if large_effects:
            recommendations.append(
                f"🎯 Large effect sizes found: {large_effects}. "
                "These differences are practically significant."
            )

        # Multiple comparisons
        if len(analysis["systems"]) > 2:
            recommendations.append(
                "🔬 Multiple comparisons detected. Use corrected p-values "
                "(Benjamini-Hochberg FDR) to control false discovery rate."
            )

        return recommendations

    def generate_comprehensive_report(self, analysis: Dict, output_file: str = None) -> str:
        """Generate a comprehensive statistical report"""
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE STATISTICAL ANALYSIS REPORT",
            "=" * 80,
            f"Metric: {analysis['metric']}",
            f"Systems Compared: {', '.join(analysis['systems'])}",
            f"Significance Level: α = {self.alpha}",
            ""
        ]

        # Descriptive Statistics
        report_lines.extend([
            "DESCRIPTIVE STATISTICS",
            "-" * 40
        ])

        for system, stats in analysis["descriptive_stats"].items():
            report_lines.extend([
                f"\n{system.upper()}:",
                f"  n = {stats['n']}, Mean = {stats['mean']:.4f} ± {stats['std']:.4f}",
                f"  Median = {stats['median']:.4f}, IQR = [{stats['q25']:.4f}, {stats['q75']:.4f}]",
                f"  Range = [{stats['min']:.4f}, {stats['max']:.4f}]",
                f"  Skewness = {stats['skewness']:.3f}, Kurtosis = {stats['kurtosis']:.3f}"
            ])

        # Statistical Tests
        report_lines.extend([
            "\n\nSTATISTICAL TESTS",
            "-" * 40
        ])

        # Add test results
        for test_type in ["parametric_tests", "non_parametric_tests"]:
            tests = analysis.get(test_type, {})
            if tests:
                report_lines.append(f"\n{test_type.replace('_', ' ').title()}:")
                for test_name, result in tests.items():
                    significance = "✓ Significant" if result.get("significant") else "✗ Not significant"
                    report_lines.append(
                        f"  {test_name}: statistic = {result['statistic']:.4f}, "
                        f"p = {result['p_value']:.4f} ({significance})"
                    )

        # Effect Sizes
        if analysis.get("effect_sizes"):
            report_lines.extend([
                "\n\nEFFECT SIZES",
                "-" * 40
            ])

            for comparison, effects in analysis["effect_sizes"].items():
                report_lines.extend([
                    f"\n{comparison.replace('_vs_', ' vs. ').title()}:",
                    f"  Cohen's d = {effects['cohens_d']:.3f} ({effects['cohens_d_interpretation']})",
                    f"  Cliff's δ = {effects['cliffs_delta']:.3f} ({effects['cliffs_delta_interpretation']})"
                ])

        # Post-hoc Tests
        if analysis.get("post_hoc_tests"):
            report_lines.extend([
                "\n\nPOST-HOC PAIRWISE COMPARISONS",
                "-" * 40
            ])

            for comparison, result in analysis["post_hoc_tests"].items():
                bh_sig = "✓" if result.get("significant_bh") else "✗"
                bonf_sig = "✓" if result.get("significant_bonferroni") else "✗"
                report_lines.append(
                    f"{comparison}: p_raw = {result['p_value_raw']:.4f}, "
                    f"p_BH = {result['p_value_bh_corrected']:.4f} ({bh_sig}), "
                    f"p_Bonf = {result['p_value_bonferroni_corrected']:.4f} ({bonf_sig})"
                )

        # Recommendations
        if analysis.get("recommendations"):
            report_lines.extend([
                "\n\nRECOMMENDATIONS",
                "-" * 40
            ])
            for rec in analysis["recommendations"]:
                report_lines.append(f"• {rec}")

        report_lines.extend(["", "=" * 80])

        report_text = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

        return report_text


# Usage Example Integration
class EnhancedQuantitativeEvaluator(RecommendationEvaluator):
    """Enhanced evaluator with advanced statistics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_analyzer = AdvancedStatisticalAnalyzer()

    def compare_systems_advanced(self,
                                 results_files: List[str],
                                 system_names: List[str],
                                 k: int = 3) -> Dict:
        """Enhanced system comparison with advanced statistics"""

        # Get basic evaluation results
        system_metrics = {}
        for file_path, system_name in zip(results_files, system_names):
            metrics = self.evaluate_recommendations(file_path, k)
            if metrics:
                system_metrics[system_name] = metrics

        if len(system_metrics) < 2:
            return {"error": "Need at least 2 systems for comparison"}

        # Advanced statistical analysis for each metric
        advanced_analyses = {}

        metrics_to_analyze = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity', 'novelty']

        for metric in metrics_to_analyze:
            system_scores = {}
            for system_name, metrics in system_metrics.items():
                scores = metrics['overall_metrics'][metric]['scores']
                system_scores[system_name] = scores

            if all(len(scores) > 0 for scores in system_scores.values()):
                analysis = self.stats_analyzer.comprehensive_comparison(
                    system_scores, metric_name=metric
                )
                advanced_analyses[metric] = analysis

        return {
            "system_metrics": system_metrics,
            "advanced_statistical_analyses": advanced_analyses,
            "comparison_info": {
                "systems_compared": system_names,
                "k": k,
                "timestamp": datetime.now().isoformat()
            }
        }

    def generate_advanced_comparison_report(self,
                                            analysis_results: Dict,
                                            output_dir: str) -> None:
        """Generate advanced comparison reports"""

        os.makedirs(output_dir, exist_ok=True)

        # Generate individual metric reports
        for metric, analysis in analysis_results["advanced_statistical_analyses"].items():
            report_file = os.path.join(output_dir, f"{metric}_statistical_analysis.txt")
            self.stats_analyzer.generate_comprehensive_report(analysis, report_file)

        # Generate summary report
        self._generate_summary_statistical_report(analysis_results, output_dir)

    def _generate_summary_statistical_report(self, analysis_results: Dict, output_dir: str):
        """Generate summary report across all metrics"""

        summary_lines = [
            "=" * 80,
            "MULTI-METRIC STATISTICAL COMPARISON SUMMARY",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Systems: {', '.join(analysis_results['comparison_info']['systems_compared'])}",
            ""
        ]

        # Overall winner analysis
        metric_winners = {}
        significant_differences = {}

        for metric, analysis in analysis_results["advanced_statistical_analyses"].items():
            # Determine winner based on means
            best_system = max(
                analysis["descriptive_stats"].items(),
                key=lambda x: x[1]["mean"]
            )[0]
            metric_winners[metric] = best_system

            # Count significant differences
            sig_count = 0
            for test_type in ["parametric_tests", "non_parametric_tests"]:
                tests = analysis.get(test_type, {})
                sig_count += sum(1 for test in tests.values() if test.get("significant", False))

            significant_differences[metric] = sig_count > 0

        # Summary statistics
        summary_lines.extend([
            "PERFORMANCE WINNERS BY METRIC",
            "-" * 40
        ])

        for metric, winner in metric_winners.items():
            significance = "✓ Significant" if significant_differences[metric] else "✗ Not significant"
            summary_lines.append(f"{metric}: {winner} ({significance})")

        # Overall system ranking
        summary_lines.extend([
            "\nOVERALL SYSTEM RANKING",
            "-" * 40
        ])

        system_wins = {}
        for system in analysis_results["comparison_info"]["systems_compared"]:
            wins = sum(1 for winner in metric_winners.values() if winner == system)
            system_wins[system] = wins

        ranked_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)
        for rank, (system, wins) in enumerate(ranked_systems, 1):
            summary_lines.append(f"{rank}. {system}: {wins}/{len(metric_winners)} metrics won")

        # Effect size summary
        summary_lines.extend([
            "\nLARGE EFFECT SIZES DETECTED",
            "-" * 40
        ])

        large_effects = []
        for metric, analysis in analysis_results["advanced_statistical_analyses"].items():
            for comparison, effects in analysis.get("effect_sizes", {}).items():
                if effects.get("cohens_d_interpretation") == "large":
                    large_effects.append(f"{metric}: {comparison}")

        if large_effects:
            for effect in large_effects:
                summary_lines.append(f"• {effect}")
        else:
            summary_lines.append("• No large effect sizes detected")

        summary_lines.extend(["", "=" * 80])

        # Save summary report
        summary_file = os.path.join(output_dir, "statistical_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_lines))