# experiments/advanced_experimental_designs.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
import json
from datetime import datetime


@dataclass
class ExperimentalCondition:
    """Defines an experimental condition."""
    name: str
    temperature: float
    prompt_template: str
    user_type: str  # 'cold', 'warm', 'mixed'
    domain_pair: str
    sample_size: int
    description: str


class AdvancedExperimentalDesigner:
    """Design and execute advanced experimental studies."""

    def __init__(self, base_tester):
        self.base_tester = base_tester
        self.experiments = {}

    def design_factorial_experiment(self) -> List[ExperimentalCondition]:
        """Design a full factorial experiment across key variables."""

        # Experimental factors
        temperatures = [0.3, 0.7, 1.0]  # Conservative, balanced, creative
        user_types = ['cold', 'warm']
        domain_pairs = ['Books_to_Movies_and_TV', 'Movies_and_TV_to_Books',
                        'Digital_Music_to_Books', 'Books_to_Digital_Music']

        # Prompt variations
        prompt_templates = {
            'standard': """Based on the user's preferences in {source_domain}, generate personalized recommendations for items from {target_domain}.

User's highly-rated items: {user_history}
Target domain: {target_domain}

Recommend top 3 items with explanations.""",

            'reasoning': """You are an expert recommendation system. Analyze the user's preferences step-by-step.

Step 1: Analyze patterns in user's {source_domain} preferences: {user_history}
Step 2: Identify transferable preferences to {target_domain}
Step 3: Recommend 3 {target_domain} items that match these patterns

Provide detailed reasoning for each recommendation.""",

            'persona': """You are a knowledgeable curator who understands cross-domain preferences.

The user loves these {source_domain} items: {user_history}

As someone who knows both {source_domain} and {target_domain} well, recommend 3 {target_domain} items this user would enjoy. Explain why each recommendation connects to their demonstrated tastes."""
        }

        conditions = []
        condition_id = 0

        for temp in temperatures:
            for user_type in user_types:
                for domain_pair in domain_pairs:
                    for template_name, template in prompt_templates.items():
                        condition = ExperimentalCondition(
                            name=f"T{temp}_U{user_type}_D{domain_pair.split('_to_')[0][:3]}_P{template_name}",
                            temperature=temp,
                            prompt_template=template,
                            user_type=user_type,
                            domain_pair=domain_pair,
                            sample_size=20,  # Users per condition
                            description=f"Temperature {temp}, {user_type} users, {domain_pair}, {template_name} prompt"
                        )
                        conditions.append(condition)
                        condition_id += 1

        return conditions

    def design_ablation_study(self) -> List[ExperimentalCondition]:
        """Design ablation study to understand component contributions."""

        base_prompt = """Based on the user's preferences in {source_domain}, recommend 3 {target_domain} items."""

        ablation_conditions = [
            # Component additions
            ExperimentalCondition(
                name="base",
                temperature=0.7,
                prompt_template=base_prompt,
                user_type="mixed",
                domain_pair="Books_to_Movies_and_TV",
                sample_size=50,
                description="Minimal prompt baseline"
            ),

            ExperimentalCondition(
                name="add_user_history",
                temperature=0.7,
                prompt_template=base_prompt + "\n\nUser's liked items: {user_history}",
                user_type="mixed",
                domain_pair="Books_to_Movies_and_TV",
                sample_size=50,
                description="Added user history"
            ),

            ExperimentalCondition(
                name="add_explanations",
                temperature=0.7,
                prompt_template=base_prompt + "\n\nUser's liked items: {user_history}\n\nProvide explanations for each recommendation.",
                user_type="mixed",
                domain_pair="Books_to_Movies_and_TV",
                sample_size=50,
                description="Added explanation requirement"
            ),

            ExperimentalCondition(
                name="add_reasoning",
                temperature=0.7,
                prompt_template=base_prompt + "\n\nUser's liked items: {user_history}\n\nFirst analyze the user's preferences, then recommend 3 items with explanations.",
                user_type="mixed",
                domain_pair="Books_to_Movies_and_TV",
                sample_size=50,
                description="Added reasoning step"
            ),

            ExperimentalCondition(
                name="full_prompt",
                temperature=0.7,
                prompt_template="""You are an expert cross-domain recommendation system.

Step 1: Analyze user's {source_domain} preferences: {user_history}
Step 2: Identify patterns that transfer to {target_domain}
Step 3: Recommend 3 {target_domain} items with detailed explanations

Output format:
1. [Item] - [Detailed explanation linking to user preferences]
2. [Item] - [Detailed explanation linking to user preferences]  
3. [Item] - [Detailed explanation linking to user preferences]""",
                user_type="mixed",
                domain_pair="Books_to_Movies_and_TV",
                sample_size=50,
                description="Full optimized prompt"
            )
        ]

        return ablation_conditions

    def design_cross_domain_transfer_study(self) -> List[ExperimentalCondition]:
        """Study transfer effectiveness across different domain pairs."""

        # Domain similarity matrix (subjective, could be computed)
        domain_similarities = {
            ('Books', 'Movies_and_TV'): 'high',  # Both narrative content
            ('Digital_Music', 'CDs'): 'high',  # Same medium
            ('Books', 'Digital_Music'): 'low',  # Different mediums
            ('Movies_and_TV', 'CDs'): 'medium'  # Both entertainment
        }

        conditions = []
        for (source, target), similarity in domain_similarities.items():
            for user_type in ['cold', 'warm']:
                condition = ExperimentalCondition(
                    name=f"transfer_{source[:3]}_{target[:3]}_{similarity}_{user_type}",
                    temperature=0.7,
                    prompt_template="""Analyze the user's {source_domain} preferences and recommend 3 {target_domain} items.

User's preferences: {user_history}

Consider how preferences might transfer between these domains.""",
                    user_type=user_type,
                    domain_pair=f"{source}_to_{target}",
                    sample_size=30,
                    description=f"Transfer study: {source} -> {target} ({similarity} similarity), {user_type} users"
                )
                conditions.append(condition)

        return conditions

    def run_experimental_condition(self, condition: ExperimentalCondition) -> Dict:
        """Execute a single experimental condition."""
        print(f"\n🧪 Running condition: {condition.name}")
        print(f"📝 {condition.description}")

        # Generate prompts for this condition
        prompts = self.base_tester.generate_test_prompts(
            user_type=condition.user_type,
            max_prompts=condition.sample_size,
            domain_pair=condition.domain_pair
        )

        if not prompts:
            return {"error": f"No prompts generated for condition {condition.name}"}

        # Modify prompts to use the experimental template
        for prompt in prompts:
            # Replace template in the prompt
            prompt['input'] = condition.prompt_template.format(
                source_domain=prompt['source_domain'],
                target_domain=prompt['target_domain'],
                user_history=prompt['input'].split('User\'s highly-rated items from')[1].split('\n')[
                    0] if 'User\'s highly-rated items from' in prompt['input'] else ""
            )

        # Run with specified temperature
        results = self.base_tester.test_prompts_with_ollama(
            prompts,
            temperature=condition.temperature,
            save_intermediate=False
        )

        # Calculate condition-specific metrics
        successful_results = [r for r in results if r['success']]

        if not successful_results:
            return {"error": f"No successful results for condition {condition.name}"}

        metrics = {
            'condition_name': condition.name,
            'success_rate': len(successful_results) / len(results),
            'avg_response_time': np.mean([r['response_time'] for r in successful_results]),
            'avg_quality': np.mean([self.base_tester.estimate_response_quality(r['ollama_response'])
                                    for r in successful_results]),
            'sample_size': len(successful_results),
            'temperature': condition.temperature,
            'user_type': condition.user_type,
            'domain_pair': condition.domain_pair,
            'results': results
        }

        return metrics

    def run_full_experiment_suite(self, experiment_type: str = 'factorial') -> Dict:
        """Run a complete experimental suite."""

        if experiment_type == 'factorial':
            conditions = self.design_factorial_experiment()
        elif experiment_type == 'ablation':
            conditions = self.design_ablation_study()
        elif experiment_type == 'transfer':
            conditions = self.design_cross_domain_transfer_study()
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        print(f"🔬 Running {experiment_type} experiment with {len(conditions)} conditions")

        experiment_results = {
            'experiment_info': {
                'type': experiment_type,
                'start_time': datetime.now().isoformat(),
                'total_conditions': len(conditions),
                'model': self.base_tester.ollama_model
            },
            'conditions': {},
            'summary': {}
        }

        # Run each condition
        for i, condition in enumerate(conditions):
            print(f"\n🔄 Progress: {i + 1}/{len(conditions)}")

            condition_results = self.run_experimental_condition(condition)
            experiment_results['conditions'][condition.name] = condition_results

            # Save intermediate results
            if (i + 1) % 5 == 0:
                self.save_experiment_results(experiment_results, f"intermediate_{experiment_type}_{i + 1}.json")

        # Generate summary
        experiment_results['summary'] = self.analyze_experiment_results(experiment_results)
        experiment_results['experiment_info']['end_time'] = datetime.now().isoformat()

        # Save final results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_experiment_results(experiment_results, f"{experiment_type}_experiment_{timestamp}.json")

        # Generate report
        self.generate_experiment_report(experiment_results, f"{experiment_type}_report_{timestamp}.txt")

        return experiment_results

    def analyze_experiment_results(self, experiment_results: Dict) -> Dict:
        """Analyze experimental results for insights."""

        conditions = experiment_results['conditions']
        valid_conditions = {name: cond for name, cond in conditions.items()
                            if 'error' not in cond}

        if not valid_conditions:
            return {'error': 'No valid conditions to analyze'}

        # Overall statistics
        all_quality_scores = []
        all_response_times = []

        for cond in valid_conditions.values():
            all_quality_scores.append(cond['avg_quality'])
            all_response_times.append(cond['avg_response_time'])

        summary = {
            'overall': {
                'valid_conditions': len(valid_conditions),
                'avg_quality': np.mean(all_quality_scores),
                'std_quality': np.std(all_quality_scores),
                'avg_response_time': np.mean(all_response_times),
                'std_response_time': np.std(all_response_times)
            }
        }

        # Factor analysis (if factorial experiment)
        if 'T0.3' in list(valid_conditions.keys())[0]:  # Factorial experiment
            summary['factor_analysis'] = self.analyze_factorial_factors(valid_conditions)

        # Best performing conditions
        sorted_conditions = sorted(valid_conditions.items(),
                                   key=lambda x: x[1]['avg_quality'],
                                   reverse=True)

        summary['top_conditions'] = [
            {
                'name': name,
                'quality': cond['avg_quality'],
                'success_rate': cond['success_rate'],
                'description': cond.get('description', '')
            }
            for name, cond in sorted_conditions[:5]
        ]

        return summary

    def analyze_factorial_factors(self, conditions: Dict) -> Dict:
        """Analyze main effects and interactions in factorial experiment."""

        # Group by factors
        by_temperature = {}
        by_user_type = {}
        by_prompt_type = {}

        for name, cond in conditions.items():
            # Extract factors from condition name (e.g., "T0.7_Ucold_DBoo_Pstandard")
            parts = name.split('_')
            temp = parts[0][1:]  # Remove 'T'
            user_type = parts[1][1:]  # Remove 'U'
            prompt_type = parts[3][1:]  # Remove 'P'

            # Group by temperature
            if temp not in by_temperature:
                by_temperature[temp] = []
            by_temperature[temp].append(cond['avg_quality'])

            # Group by user type
            if user_type not in by_user_type:
                by_user_type[user_type] = []
            by_user_type[user_type].append(cond['avg_quality'])

            # Group by prompt type
            if prompt_type not in by_prompt_type:
                by_prompt_type[prompt_type] = []
            by_prompt_type[prompt_type].append(cond['avg_quality'])

        # Calculate main effects
        factor_analysis = {
            'temperature_effects': {
                temp: np.mean(scores) for temp, scores in by_temperature.items()
            },
            'user_type_effects': {
                user_type: np.mean(scores) for user_type, scores in by_user_type.items()
            },
            'prompt_type_effects': {
                prompt_type: np.mean(scores) for prompt_type, scores in by_prompt_type.items()
            }
        }

        # Identify best levels for each factor
        factor_analysis['best_temperature'] = max(factor_analysis['temperature_effects'].items(),
                                                  key=lambda x: x[1])
        factor_analysis['best_user_type'] = max(factor_analysis['user_type_effects'].items(),
                                                key=lambda x: x[1])
        factor_analysis['best_prompt_type'] = max(factor_analysis['prompt_type_effects'].items(),
                                                  key=lambda x: x[1])

        return factor_analysis

    def generate_experiment_report(self, experiment_results: Dict, output_file: str):
        """Generate comprehensive experimental report."""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"ADVANCED EXPERIMENTAL ANALYSIS REPORT")
        report_lines.append("=" * 80)

        info = experiment_results['experiment_info']
        summary = experiment_results['summary']

        report_lines.append(f"Experiment Type: {info['type'].upper()}")
        report_lines.append(f"Start Time: {info['start_time']}")
        report_lines.append(f"End Time: {info.get('end_time', 'Running...')}")
        report_lines.append(f"Model: {info['model']}")
        report_lines.append(f"Total Conditions: {info['total_conditions']}")
        report_lines.append("")

        if 'error' not in summary:
            overall = summary['overall']
            report_lines.append("OVERALL RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Valid Conditions: {overall['valid_conditions']}")
            report_lines.append(f"Average Quality: {overall['avg_quality']:.3f} ± {overall['std_quality']:.3f}")
            report_lines.append(
                f"Average Response Time: {overall['avg_response_time']:.2f}s ± {overall['std_response_time']:.2f}s")
            report_lines.append("")

            # Top performing conditions
            report_lines.append("TOP PERFORMING CONDITIONS")
            report_lines.append("-" * 40)
            for i, cond in enumerate(summary['top_conditions'], 1):
                report_lines.append(f"{i}. {cond['name']}")
                report_lines.append(f"   Quality: {cond['quality']:.3f}")
                report_lines.append(f"   Success Rate: {cond['success_rate']:.1%}")
                report_lines.append("")

            # Factor analysis (if available)
            if 'factor_analysis' in summary:
                fa = summary['factor_analysis']
                report_lines.append("FACTOR ANALYSIS")
                report_lines.append("-" * 40)

                report_lines.append("Temperature Effects:")
                for temp, effect in fa['temperature_effects'].items():
                    report_lines.append(f"  {temp}: {effect:.3f}")
                report_lines.append(f"  Best: {fa['best_temperature'][0]} ({fa['best_temperature'][1]:.3f})")
                report_lines.append("")

                report_lines.append("User Type Effects:")
                for user_type, effect in fa['user_type_effects'].items():
                    report_lines.append(f"  {user_type}: {effect:.3f}")
                report_lines.append(f"  Best: {fa['best_user_type'][0]} ({fa['best_user_type'][1]:.3f})")
                report_lines.append("")

                report_lines.append("Prompt Type Effects:")
                for prompt_type, effect in fa['prompt_type_effects'].items():
                    report_lines.append(f"  {prompt_type}: {effect:.3f}")
                report_lines.append(f"  Best: {fa['best_prompt_type'][0]} ({fa['best_prompt_type'][1]:.3f})")
                report_lines.append("")

        report_lines.append("=" * 80)

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"📋 Experimental report saved to: {output_file}")

    def save_experiment_results(self, results: Dict, filename: str):
        """Save experimental results to JSON."""
        import os
        os.makedirs("results/experiments", exist_ok=True)

        filepath = f"results/experiments/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Experiment results saved to: {filepath}")


# Example usage
def run_complete_experimental_suite():
    """Example of running complete experimental analysis."""

    # Initialize base tester
    from test_prompt_generator_cold_warm import PromptGeneratorTester
    base_tester = PromptGeneratorTester()

    # Create experimental designer
    designer = AdvancedExperimentalDesigner(base_tester)

    # Run different experiment types
    experiments = ['factorial', 'ablation', 'transfer']

    all_results = {}

    for exp_type in experiments:
        print(f"\n🚀 Starting {exp_type} experiment...")
        results = designer.run_full_experiment_suite(exp_type)
        all_results[exp_type] = results

        # Print quick summary
        if 'summary' in results and 'error' not in results['summary']:
            summary = results['summary']
            print(f"✅ {exp_type} completed:")
            print(f"   Valid conditions: {summary['overall']['valid_conditions']}")
            print(f"   Avg quality: {summary['overall']['avg_quality']:.3f}")

            if 'top_conditions' in summary:
                best_condition = summary['top_conditions'][0]
                print(f"   Best condition: {best_condition['name']} ({best_condition['quality']:.3f})")

    return all_results


# Statistical analysis utilities
class ExperimentalStatistics:
    """Statistical analysis tools for experimental results."""

    @staticmethod
    def compute_effect_size(group1_scores: List[float],
                            group2_scores: List[float]) -> float:
        """Compute Cohen's d effect size."""
        mean1, mean2 = np.mean(group1_scores), np.mean(group2_scores)
        std1, std2 = np.std(group1_scores, ddof=1), np.std(group2_scores, ddof=1)

        # Pooled standard deviation
        n1, n2 = len(group1_scores), len(group2_scores)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    @staticmethod
    def interpret_effect_size(effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    @staticmethod
    def multiple_comparison_correction(p_values: List[float],
                                       method: str = 'bonferroni') -> List[float]:
        """Apply multiple comparison correction."""
        if method == 'bonferroni':
            return [min(1.0, p * len(p_values)) for p in p_values]
        elif method == 'fdr':  # False Discovery Rate (Benjamini-Hochberg)
            sorted_pvals = sorted(enumerate(p_values), key=lambda x: x[1])
            corrected = [0.0] * len(p_values)

            for i, (orig_idx, p_val) in enumerate(sorted_pvals):
                corrected[orig_idx] = min(1.0, p_val * len(p_values) / (i + 1))

            return corrected
        else:
            raise ValueError(f"Unknown correction method: {method}")

    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05,
                       power: float = 0.8) -> int:
        """Estimate required sample size for given effect size and power."""
        # Simplified power analysis for two-sample t-test
        # More sophisticated analysis would use scipy.stats or statsmodels

        if effect_size == 0:
            return float('inf')

        # Rough approximation
        z_alpha = 1.96  # for alpha = 0.05
        z_beta = 0.84  # for power = 0.8

        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n_per_group))


# Meta-analysis across experiments
class MetaAnalyzer:
    """Conduct meta-analysis across multiple experiments."""

    def __init__(self):
        self.experiments = {}

    def add_experiment(self, name: str, results: Dict):
        """Add experimental results for meta-analysis."""
        self.experiments[name] = results

    def meta_analyze_factors(self) -> Dict:
        """Conduct meta-analysis of experimental factors."""

        # Collect effect sizes across experiments
        temperature_effects = []
        user_type_effects = []
        prompt_effects = []

        for exp_name, exp_results in self.experiments.items():
            if ('conditions' in exp_results and
                    'summary' in exp_results and
                    'factor_analysis' in exp_results['summary']):

                fa = exp_results['summary']['factor_analysis']

                # Temperature effects
                temp_scores = list(fa['temperature_effects'].values())
                if len(temp_scores) > 1:
                    temp_effect = max(temp_scores) - min(temp_scores)
                    temperature_effects.append(temp_effect)

                # User type effects
                user_scores = list(fa['user_type_effects'].values())
                if len(user_scores) > 1:
                    user_effect = max(user_scores) - min(user_scores)
                    user_type_effects.append(user_effect)

                # Prompt effects
                prompt_scores = list(fa['prompt_type_effects'].values())
                if len(prompt_scores) > 1:
                    prompt_effect = max(prompt_scores) - min(prompt_scores)
                    prompt_effects.append(prompt_effect)

        meta_results = {}

        if temperature_effects:
            meta_results['temperature'] = {
                'mean_effect': np.mean(temperature_effects),
                'std_effect': np.std(temperature_effects),
                'n_experiments': len(temperature_effects)
            }

        if user_type_effects:
            meta_results['user_type'] = {
                'mean_effect': np.mean(user_type_effects),
                'std_effect': np.std(user_type_effects),
                'n_experiments': len(user_type_effects)
            }

        if prompt_effects:
            meta_results['prompt_type'] = {
                'mean_effect': np.mean(prompt_effects),
                'std_effect': np.std(prompt_effects),
                'n_experiments': len(prompt_effects)
            }

        return meta_results

    def identify_robust_findings(self, consistency_threshold: float = 0.7) -> Dict:
        """Identify findings that are consistent across experiments."""

        # Collect top conditions across experiments
        top_conditions_by_experiment = {}

        for exp_name, exp_results in self.experiments.items():
            if ('summary' in exp_results and
                    'top_conditions' in exp_results['summary']):
                top_conditions = exp_results['summary']['top_conditions']
                top_conditions_by_experiment[exp_name] = top_conditions

        # Find consistently high-performing factors
        factor_counts = {
            'temperature': {},
            'user_type': {},
            'prompt_type': {}
        }

        total_experiments = len(top_conditions_by_experiment)

        for exp_name, top_conditions in top_conditions_by_experiment.items():
            # Look at top 3 conditions from each experiment
            for condition in top_conditions[:3]:
                name = condition['name']

                # Extract factors (assuming naming convention)
                if '_T' in name:
                    temp = name.split('_T')[1].split('_')[0]
                    factor_counts['temperature'][temp] = factor_counts['temperature'].get(temp, 0) + 1

                if '_U' in name:
                    user_type = name.split('_U')[1].split('_')[0]
                    factor_counts['user_type'][user_type] = factor_counts['user_type'].get(user_type, 0) + 1

                if '_P' in name:
                    prompt_type = name.split('_P')[1].split('_')[0]
                    factor_counts['prompt_type'][prompt_type] = factor_counts['prompt_type'].get(prompt_type, 0) + 1

        # Identify robust factors (appear in most experiments)
        robust_findings = {}

        for factor_type, counts in factor_counts.items():
            for factor_value, count in counts.items():
                consistency = count / total_experiments
                if consistency >= consistency_threshold:
                    if factor_type not in robust_findings:
                        robust_findings[factor_type] = []

                    robust_findings[factor_type].append({
                        'value': factor_value,
                        'consistency': consistency,
                        'appeared_in': count,
                        'total_experiments': total_experiments
                    })

        return robust_findings

    def generate_meta_analysis_report(self, output_file: str):
        """Generate comprehensive meta-analysis report."""

        meta_effects = self.meta_analyze_factors()
        robust_findings = self.identify_robust_findings()

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("META-ANALYSIS REPORT")
        report_lines.append("=" * 80)

        report_lines.append(f"Experiments Analyzed: {len(self.experiments)}")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Meta-analysis results
        report_lines.append("META-ANALYTIC EFFECTS")
        report_lines.append("-" * 40)

        for factor, effects in meta_effects.items():
            report_lines.append(f"{factor.upper()}:")
            report_lines.append(f"  Mean Effect Size: {effects['mean_effect']:.3f}")
            report_lines.append(f"  Standard Deviation: {effects['std_effect']:.3f}")
            report_lines.append(f"  Number of Studies: {effects['n_experiments']}")
            report_lines.append("")

        # Robust findings
        report_lines.append("ROBUST FINDINGS (>70% consistency)")
        report_lines.append("-" * 40)

        for factor_type, findings in robust_findings.items():
            if findings:
                report_lines.append(f"{factor_type.upper()}:")
                for finding in findings:
                    report_lines.append(f"  {finding['value']}: {finding['consistency']:.1%} consistency "
                                        f"({finding['appeared_in']}/{finding['total_experiments']} experiments)")
                report_lines.append("")

        # Recommendations
        report_lines.append("EVIDENCE-BASED RECOMMENDATIONS")
        report_lines.append("-" * 40)

        # Temperature recommendations
        if 'temperature' in robust_findings:
            best_temp = max(robust_findings['temperature'], key=lambda x: x['consistency'])
            report_lines.append(f"• Use temperature {best_temp['value']} "
                                f"(consistent in {best_temp['consistency']:.1%} of studies)")

        # User type insights
        if 'user_type' in robust_findings:
            best_user_type = max(robust_findings['user_type'], key=lambda x: x['consistency'])
            report_lines.append(f"• {best_user_type['value']} users show better performance "
                                f"(consistent in {best_user_type['consistency']:.1%} of studies)")

        # Prompt recommendations
        if 'prompt_type' in robust_findings:
            best_prompt = max(robust_findings['prompt_type'], key=lambda x: x['consistency'])
            report_lines.append(f"• Use {best_prompt['value']} prompting approach "
                                f"(consistent in {best_prompt['consistency']:.1%} of studies)")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"📊 Meta-analysis report saved to: {output_file}")


# Example of complete experimental pipeline
def run_comprehensive_experimental_analysis():
    """Run complete experimental analysis pipeline."""

    print("🔬 COMPREHENSIVE EXPERIMENTAL ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Run experimental suite
    results = run_complete_experimental_suite()

    # 2. Conduct meta-analysis
    meta_analyzer = MetaAnalyzer()

    for exp_type, exp_results in results.items():
        meta_analyzer.add_experiment(exp_type, exp_results)

    # 3. Generate meta-analysis report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    meta_analyzer.generate_meta_analysis_report(f"results/meta_analysis_report_{timestamp}.txt")

    # 4. Statistical analysis
    stats_analyzer = ExperimentalStatistics()

    # Example: Compare cold vs warm users across all experiments
    cold_scores = []
    warm_scores = []

    for exp_results in results.values():
        if 'conditions' in exp_results:
            for cond_name, cond_data in exp_results['conditions'].items():
                if 'error' not in cond_data:
                    if 'cold' in cond_name:
                        cold_scores.append(cond_data['avg_quality'])
                    elif 'warm' in cond_name:
                        warm_scores.append(cond_data['avg_quality'])

    if cold_scores and warm_scores:
        effect_size = stats_analyzer.compute_effect_size(warm_scores, cold_scores)
        interpretation = stats_analyzer.interpret_effect_size(effect_size)

        print(f"\n📈 COLD vs WARM USER ANALYSIS:")
        print(f"   Effect size (Cohen's d): {effect_size:.3f}")
        print(f"   Interpretation: {interpretation}")
        print(f"   Warm users avg: {np.mean(warm_scores):.3f}")
        print(f"   Cold users avg: {np.mean(cold_scores):.3f}")

    print("\n✅ Comprehensive experimental analysis completed!")
    return results, meta_analyzer


if __name__ == "__main__":
    # Run comprehensive analysis
    results, meta_analyzer = run_comprehensive_experimental_analysis()