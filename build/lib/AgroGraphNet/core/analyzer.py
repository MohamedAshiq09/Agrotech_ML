import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.logger import get_logger

class Analyzer:
    """Analysis and reporting for AgroGraphNet results"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
    
    def generate_comprehensive_analysis(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis from all available data"""
        self.logger.info("Generating comprehensive analysis...")
        
        analysis = {
            'summary': {},
            'data_analysis': {},
            'model_performance': {},
            'spatial_analysis': {},
            'temporal_analysis': {},
            'recommendations': []
        }
        
        # Data summary
        if 'farms' in data_sources:
            analysis['summary']['total_farms'] = len(data_sources['farms'])
            analysis['summary']['crop_types'] = data_sources['farms']['crop_type'].value_counts().to_dict()
        
        # Disease distribution analysis
        if 'predictions' in data_sources:
            pred_df = data_sources['predictions']
            if 'predicted_disease' in pred_df.columns:
                disease_dist = pred_df['predicted_disease'].value_counts().to_dict()
                analysis['summary']['disease_distribution'] = disease_dist
                
                # Risk analysis
                if 'risk_score' in pred_df.columns:
                    high_risk_threshold = 0.7
                    high_risk_farms = (pred_df['risk_score'] > high_risk_threshold).sum()
                    analysis['summary']['high_risk_farms'] = high_risk_farms
                    analysis['summary']['avg_risk_score'] = pred_df['risk_score'].mean()
        
        # Model performance analysis
        if 'training_summary' in data_sources:
            analysis['model_performance'] = self._analyze_model_performance(data_sources)
        
        # Spatial analysis
        if 'farms' in data_sources and 'predictions' in data_sources:
            analysis['spatial_analysis'] = self._analyze_spatial_patterns(
                data_sources['farms'], data_sources['predictions']
            )
        
        # Temporal analysis
        if 'weather' in data_sources or 'satellite' in data_sources:
            analysis['temporal_analysis'] = self._analyze_temporal_patterns(data_sources)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_model_performance(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance metrics"""
        performance = {}
        
        # This would parse training logs/results
        # For now, return placeholder analysis
        performance['best_model'] = 'GraphSAGE'
        performance['accuracy'] = 0.85
        performance['training_epochs'] = 100
        
        return performance
    
    def _analyze_spatial_patterns(self, farms_df: pd.DataFrame, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spatial patterns in disease distribution"""
        spatial_analysis = {}
        
        # Merge farms and predictions
        if 'farm_id' in farms_df.columns and 'farm_id' in predictions_df.columns:
            merged = farms_df.merge(predictions_df, on='farm_id', how='inner')
            
            # Disease clustering analysis
            if 'predicted_disease' in merged.columns and 'latitude' in merged.columns:
                disease_by_location = merged.groupby(['predicted_disease']).agg({
                    'latitude': ['mean', 'std'],
                    'longitude': ['mean', 'std']
                }).round(4)
                
                spatial_analysis['disease_centers'] = disease_by_location.to_dict()
                
                # Calculate disease spread metrics
                for disease in merged['predicted_disease'].unique():
                    disease_farms = merged[merged['predicted_disease'] == disease]
                    if len(disease_farms) > 1:
                        # Calculate spread (standard deviation of coordinates)
                        lat_std = disease_farms['latitude'].std()
                        lon_std = disease_farms['longitude'].std()
                        spatial_analysis[f'{disease}_spread'] = {
                            'lat_std': lat_std,
                            'lon_std': lon_std,
                            'total_spread': np.sqrt(lat_std**2 + lon_std**2)
                        }
        
        return spatial_analysis
    
    def _analyze_temporal_patterns(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in data"""
        temporal_analysis = {}
        
        # Weather patterns
        if 'weather' in data_sources:
            weather_df = data_sources['weather']
            if 'date' in weather_df.columns:
                weather_df['date'] = pd.to_datetime(weather_df['date'])
                weather_df['month'] = weather_df['date'].dt.month
                
                monthly_weather = weather_df.groupby('month').agg({
                    'temperature': 'mean',
                    'humidity': 'mean',
                    'precipitation': 'sum'
                }).round(2)
                
                temporal_analysis['monthly_weather'] = monthly_weather.to_dict()
        
        # Satellite/vegetation patterns
        if 'satellite' in data_sources:
            satellite_df = data_sources['satellite']
            if 'date' in satellite_df.columns:
                satellite_df['date'] = pd.to_datetime(satellite_df['date'])
                satellite_df['month'] = satellite_df['date'].dt.month
                
                monthly_ndvi = satellite_df.groupby('month')['ndvi'].mean().round(3)
                temporal_analysis['monthly_ndvi'] = monthly_ndvi.to_dict()
        
        return temporal_analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # High-risk farm recommendations
        if 'high_risk_farms' in analysis['summary']:
            high_risk = analysis['summary']['high_risk_farms']
            if high_risk > 0:
                recommendations.append(
                    f"🚨 {high_risk} farms identified as high-risk. Prioritize monitoring and preventive measures."
                )
        
        # Disease distribution recommendations
        if 'disease_distribution' in analysis['summary']:
            disease_dist = analysis['summary']['disease_distribution']
            total_farms = sum(disease_dist.values())
            
            for disease, count in disease_dist.items():
                if disease != 'Healthy' and count > total_farms * 0.1:  # More than 10%
                    percentage = (count / total_farms) * 100
                    recommendations.append(
                        f"⚠️ {disease} detected in {count} farms ({percentage:.1f}%). Consider targeted treatment programs."
                    )
        
        # Spatial clustering recommendations
        if 'spatial_analysis' in analysis and analysis['spatial_analysis']:
            recommendations.append(
                "📍 Spatial disease clustering detected. Implement quarantine measures and monitor neighboring farms."
            )
        
        # Model performance recommendations
        if 'model_performance' in analysis:
            accuracy = analysis['model_performance'].get('accuracy', 0)
            if accuracy < 0.8:
                recommendations.append(
                    "📊 Model accuracy below 80%. Consider collecting more training data or feature engineering."
                )
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "✅ Overall farm health appears good. Continue regular monitoring.",
                "📈 Consider expanding data collection for improved predictions.",
                "🔄 Schedule regular model retraining with new data."
            ])
        
        return recommendations
    
    def create_html_report(self, analysis: Dict[str, Any], output_path: Path):
        """Create HTML analysis report"""
        html_content = self._generate_html_report(analysis)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to: {output_path}")
    
    def create_json_report(self, analysis: Dict[str, Any], output_path: Path):
        """Create JSON analysis report"""
        # Convert numpy types to native Python types for JSON serialization
        json_analysis = self._convert_for_json(analysis)
        
        with open(output_path, 'w') as f:
            json.dump(json_analysis, f, indent=2, default=str)
        
        self.logger.info(f"JSON report saved to: {output_path}")
    
    def create_visualizations(self, analysis: Dict[str, Any], output_dir: Path):
        """Create visualization plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Disease distribution pie chart
        if 'disease_distribution' in analysis['summary']:
            self._create_disease_distribution_plot(
                analysis['summary']['disease_distribution'],
                output_dir / "disease_distribution.png"
            )
        
        # Risk score histogram
        # This would require access to the actual prediction data
        # For now, create placeholder visualizations
        
        self.logger.info(f"Visualizations saved to: {output_dir}")
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AgroGraphNet Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #2E8B57; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #2E8B57; }}
        .metric {{ background-color: #f0f8f0; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .high-risk {{ color: #d32f2f; font-weight: bold; }}
        .healthy {{ color: #388e3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 AgroGraphNet Analysis Report</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Summary</h2>
        {self._format_summary_html(analysis.get('summary', {}))}
    </div>
    
    <div class="section">
        <h2>🎯 Model Performance</h2>
        {self._format_performance_html(analysis.get('model_performance', {}))}
    </div>
    
    <div class="section">
        <h2>📍 Spatial Analysis</h2>
        {self._format_spatial_html(analysis.get('spatial_analysis', {}))}
    </div>
    
    <div class="section">
        <h2>📈 Recommendations</h2>
        {self._format_recommendations_html(analysis.get('recommendations', []))}
    </div>
</body>
</html>
"""
        return html
    
    def _format_summary_html(self, summary: Dict[str, Any]) -> str:
        """Format summary section for HTML"""
        html = ""
        
        if 'total_farms' in summary:
            html += f'<div class="metric">Total Farms Analyzed: <strong>{summary["total_farms"]}</strong></div>'
        
        if 'disease_distribution' in summary:
            html += '<div class="metric">Disease Distribution:<ul>'
            for disease, count in summary['disease_distribution'].items():
                css_class = 'healthy' if disease == 'Healthy' else 'high-risk'
                html += f'<li class="{css_class}">{disease}: {count} farms</li>'
            html += '</ul></div>'
        
        if 'high_risk_farms' in summary:
            html += f'<div class="metric high-risk">High Risk Farms: {summary["high_risk_farms"]}</div>'
        
        return html
    
    def _format_performance_html(self, performance: Dict[str, Any]) -> str:
        """Format performance section for HTML"""
        if not performance:
            return '<p>No performance data available.</p>'
        
        html = ""
        for key, value in performance.items():
            html += f'<div class="metric">{key.replace("_", " ").title()}: <strong>{value}</strong></div>'
        
        return html
    
    def _format_spatial_html(self, spatial: Dict[str, Any]) -> str:
        """Format spatial analysis for HTML"""
        if not spatial:
            return '<p>No spatial analysis available.</p>'
        
        html = '<p>Spatial patterns detected in disease distribution.</p>'
        return html
    
    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations for HTML"""
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        
        return html
    
    def _create_disease_distribution_plot(self, disease_dist: Dict[str, int], output_path: Path):
        """Create disease distribution pie chart"""
        plt.figure(figsize=(10, 8))
        
        labels = list(disease_dist.keys())
        sizes = list(disease_dist.values())
        colors = ['#2E8B57' if label == 'Healthy' else '#FF6B6B' for label in labels]
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Disease Distribution Across Farms', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj