import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('netflix_analysis.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    'data': {
        'required_columns': ['show_id', 'type', 'title', 'director', 'cast',
                           'country', 'date_added', 'release_year', 'rating',
                           'duration', 'listed_in', 'description'],
        'date_format': '%B %d, %Y',
        'null_indicator': 'Unknown'
    },
    'analysis': {
        'top_countries_limit': 10,
        'top_genres_limit': 15,
        'content_types': ['Movie', 'TV Show']
    },
    'visualization': {
        'sizes': {
            'large': (14, 10),
            'medium': (12, 8),
            'small': (10, 6)
        },
        'style': 'darkgrid',
        'palettes': {
            'type': 'Set1',
            'countries': 'viridis',
            'genres': 'magma',
            'ratings': 'coolwarm'
        },
        'dpi': 300
    },
    'output': {
        'formats': ['pdf', 'png', 'html'],
        'plot_prefix': 'netflix_analysis',
        'report_filename': 'netflix_analysis_report',
        'summary_filename': 'netflix_analysis_summary'
    }
}

class NetflixAnalysisError(Exception):
    pass

class DataLoader:
    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise NetflixAnalysisError(f"Data file not found: {file_path}")

        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)

            missing_cols = set(CONFIG['data']['required_columns']) - set(df.columns)
            if missing_cols:
                raise NetflixAnalysisError(f"Missing required columns: {missing_cols}")

            logger.info(f"Successfully loaded {len(df):,} records with {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise NetflixAnalysisError(f"Data loading failed: {e}") from e

    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Performing data quality validation")

        quality_report = {
            'total_records': len(df),
            'columns': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'missing_data': {
                'total_nulls': df.isnull().sum().sum(),
                'null_percentages': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            'data_types': df.dtypes.to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }

        if quality_report['duplicate_rows'] > 0:
            logger.warning(f"Found {quality_report['duplicate_rows']} duplicate rows")

        null_pct = quality_report['missing_data']['null_percentages']
        high_null_cols = [col for col, pct in null_pct.items() if pct > 50]
        if high_null_cols:
            logger.warning(f"Columns with >50% missing data: {high_null_cols}")

        return quality_report

class DataPreprocessor:
    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filling missing values")

        fill_values = {
            'director': CONFIG['data']['null_indicator'],
            'cast': CONFIG['data']['null_indicator'],
            'country': CONFIG['data']['null_indicator']
        }

        return df.fillna(fill_values)

    @staticmethod
    def clean_critical_columns(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing rows with missing critical data")

        original_count = len(df)
        critical_cols = ['date_added', 'rating', 'duration']
        df_clean = df.dropna(subset=critical_cols)

        removed_count = original_count - len(df_clean)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with missing critical data")

        return df_clean

    @staticmethod
    def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Parsing date columns")

        df = df.copy()
        df['date_added'] = pd.to_datetime(
            df['date_added'].str.strip(),
            format=CONFIG['data']['date_format'],
            errors='coerce'
        )

        null_dates = df['date_added'].isnull().sum()
        if null_dates > 0:
            logger.warning(f"Could not parse {null_dates} date values")

        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing duplicate rows")

        original_count = len(df)
        df_cleaned = df.drop_duplicates()

        duplicates_removed = original_count - len(df_cleaned)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

        return df_cleaned

    @staticmethod
    def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("Starting data preprocessing pipeline")

        processing_stats = {
            'original_records': len(df),
            'steps': []
        }

        df = DataPreprocessor.fill_missing_values(df)
        processing_stats['steps'].append({
            'step': 'fill_missing_values',
            'records': len(df)
        })

        df = DataPreprocessor.clean_critical_columns(df)
        processing_stats['steps'].append({
            'step': 'clean_critical_columns',
            'records': len(df)
        })

        df = DataPreprocessor.parse_dates(df)
        processing_stats['steps'].append({
            'step': 'parse_dates',
            'records': len(df)
        })

        df = DataPreprocessor.remove_duplicates(df)
        processing_stats['steps'].append({
            'step': 'remove_duplicates',
            'records': len(df)
        })

        processing_stats['final_records'] = len(df)
        processing_stats['records_removed'] = processing_stats['original_records'] - len(df)

        logger.info(f"Preprocessing complete: {processing_stats['final_records']} records retained")
        return df, processing_stats

class NetflixAnalyzer:
    @staticmethod
    def analyze_content_types(df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Analyzing content type distribution")

        content_stats = df['type'].value_counts().to_dict()
        content_pct = (df['type'].value_counts(normalize=True) * 100).round(1).to_dict()

        return {
            'counts': content_stats,
            'percentages': content_pct,
            'total_content': len(df),
            'unique_types': df['type'].nunique()
        }

    @staticmethod
    def analyze_countries(df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        logger.info(f"Analyzing top {top_n} producing countries")

        country_df = df[df['country'] != CONFIG['data']['null_indicator']].copy()
        country_df['primary_country'] = country_df['country'].apply(
            lambda x: x.split(',')[0].strip()
        )

        top_countries = country_df['primary_country'].value_counts().head(top_n)

        return {
            'top_countries': top_countries.to_dict(),
            'total_countries': country_df['primary_country'].nunique(),
            'country_data_available': len(country_df)
        }

    @staticmethod
    def analyze_content_trends(df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Analyzing content trends over time")

        df_trends = df.copy()
        df_trends['year_added'] = df_trends['date_added'].dt.year

        yearly_trends = df_trends.groupby('year_added')['type'].value_counts().unstack().fillna(0)

        growth_rates = {}
        for content_type in CONFIG['analysis']['content_types']:
            if content_type in yearly_trends.columns:
                values = yearly_trends[content_type]
                growth_rate = values.pct_change().mean() * 100
                growth_rates[content_type] = round(growth_rate, 2)

        return {
            'yearly_data': yearly_trends.to_dict(),
            'years_range': yearly_trends.index.tolist(),
            'growth_rates': growth_rates,
            'peak_years': {
                'Movie': yearly_trends.get('Movie', pd.Series()).idxmax() if 'Movie' in yearly_trends.columns else None,
                'TV Show': yearly_trends.get('TV Show', pd.Series()).idxmax() if 'TV Show' in yearly_trends.columns else None
            }
        }

    @staticmethod
    def analyze_genres(df: pd.DataFrame, top_n: int = 15) -> Dict[str, Any]:
        logger.info(f"Analyzing top {top_n} genres")

        all_genres = []
        df['listed_in'].str.split(', ').apply(lambda x: all_genres.extend(x))

        top_genres = Counter(all_genres).most_common(top_n)
        genre_df = pd.DataFrame(top_genres, columns=['genre', 'count'])

        return {
            'top_genres': dict(top_genres),
            'total_genres': len(set(all_genres)),
            'genre_distribution': genre_df.to_dict('records')
        }

    @staticmethod
    def analyze_ratings(df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Analyzing content rating distribution")

        rating_counts = df['rating'].value_counts()
        rating_pct = (rating_counts / len(df) * 100).round(1)

        maturity_groups = {
            'Kids': ['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'G', 'TV-G'],
            'Teens': ['TV-PG', 'PG', 'PG-13', 'TV-14'],
            'Adults': ['TV-MA', 'R', 'NC-17', 'NR'],
            'Unrated': ['UR']
        }

        groups = {}
        for category, ratings in maturity_groups.items():
            group_count = rating_counts[rating_counts.index.isin(ratings)].sum()
            groups[category] = int(group_count)

        return {
            'rating_counts': rating_counts.to_dict(),
            'rating_percentages': rating_pct.to_dict(),
            'unique_ratings': len(rating_counts),
            'maturity_groups': groups
        }

    @staticmethod
    def perform_complete_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting complete analysis")

        analysis_results = {
            'content_types': NetflixAnalyzer.analyze_content_types(df),
            'countries': NetflixAnalyzer.analyze_countries(df),
            'trends': NetflixAnalyzer.analyze_content_trends(df),
            'genres': NetflixAnalyzer.analyze_genres(df),
            'ratings': NetflixAnalyzer.analyze_ratings(df)
        }

        analysis_results['summary'] = {
            'total_titles': len(df),
            'date_range': {
                'earliest': df['date_added'].min().strftime('%Y-%m-%d'),
                'latest': df['date_added'].max().strftime('%Y-%m-%d')
            },
            'avg_titles_per_year': round(len(df) / df['date_added'].dt.year.nunique(), 2),
            'release_years_range': f"{df['release_year'].min()} - {df['release_year'].max()}"
        }

        logger.info("Analysis complete")
        return analysis_results

class VisualizationGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        warnings.filterwarnings('ignore')

    def _apply_plot_styling(self, ax: plt.Axes = None, title: str = "", xlabel: str = "",
                           ylabel: str = "", rotation: int = 0) -> None:
        if ax is not None:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

        if ax is None:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)

        if rotation:
            plt.xticks(rotation=rotation, ha='right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def create_content_type_plot(self, analysis_results: Dict) -> None:
        logger.info("Creating content type plot")

        content_data = analysis_results['content_types']

        plt.figure(figsize=CONFIG['visualization']['sizes']['small'])
        sns.set_style(CONFIG['visualization']['style'])

        types = list(content_data['counts'].keys())
        counts = list(content_data['counts'].values())

        ax = sns.barplot(x=types, y=counts, palette=CONFIG['visualization']['palettes']['type'])

        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5), textcoords='offset points',
                fontsize=10, fontweight='bold'
            )

        self._apply_plot_styling(ax, 'Netflix Content Type Distribution',
                               'Content Type', 'Count')

        plt.savefig(self.output_dir / '1_content_types.png',
                   dpi=CONFIG['visualization']['dpi'], bbox_inches='tight')
        plt.close()

    def create_countries_plot(self, analysis_results: Dict) -> None:
        logger.info("Creating countries plot")

        countries_data = analysis_results['countries']

        plt.figure(figsize=CONFIG['visualization']['sizes']['medium'])
        sns.set_style(CONFIG['visualization']['style'])

        countries = list(countries_data['top_countries'].keys())
        counts = list(countries_data['top_countries'].values())

        ax = sns.barplot(x=counts, y=countries,
                        palette=CONFIG['visualization']['palettes']['countries'])

        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_width())}',
                (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center',
                xytext=(5, 0), textcoords='offset points',
                fontsize=9
            )

        self._apply_plot_styling(ax, 'Top 10 Content Producing Countries',
                               'Content Count', 'Country')

        plt.savefig(self.output_dir / '2_top_countries.png',
                   dpi=CONFIG['visualization']['dpi'], bbox_inches='tight')
        plt.close()

    def create_trends_plot(self, analysis_results: Dict) -> None:
        logger.info("Creating trends plot")

        trends_data = analysis_results['trends']

        plt.figure(figsize=CONFIG['visualization']['sizes']['medium'])
        sns.set_style('whitegrid')

        years = trends_data['years_range']
        for content_type in CONFIG['analysis']['content_types']:
            if content_type in trends_data['yearly_data']:
                values = [trends_data['yearly_data'][content_type].get(year, 0) for year in years]
                plt.plot(years, values, label=content_type, marker='o', linewidth=2.5, markersize=6)

        plt.legend(fontsize=12, loc='upper left')
        plt.xticks(years[::2])

        self._apply_plot_styling(ax=None, title='Content Addition Trends Over Time',
                               xlabel='Year Added', ylabel='Content Count')

        plt.savefig(self.output_dir / '3_content_trends.png',
                   dpi=CONFIG['visualization']['dpi'], bbox_inches='tight')
        plt.close()

    def create_genres_plot(self, analysis_results: Dict) -> None:
        logger.info("Creating genres plot")

        genres_data = analysis_results['genres']

        plt.figure(figsize=CONFIG['visualization']['sizes']['large'])
        sns.set_style(CONFIG['visualization']['style'])

        genres = list(genres_data['top_genres'].keys())
        counts = list(genres_data['top_genres'].values())

        ax = sns.barplot(x=counts, y=genres,
                        palette=CONFIG['visualization']['palettes']['genres'])

        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_width())}',
                (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center',
                xytext=(5, 0), textcoords='offset points',
                fontsize=9
            )

        self._apply_plot_styling(ax, 'Top 15 Most Popular Genres',
                               'Content Count', 'Genre')

        plt.savefig(self.output_dir / '4_top_genres.png',
                   dpi=CONFIG['visualization']['dpi'], bbox_inches='tight')
        plt.close()

    def create_ratings_plot(self, analysis_results: Dict) -> None:
        logger.info("Creating ratings plot")

        ratings_data = analysis_results['ratings']

        plt.figure(figsize=CONFIG['visualization']['sizes']['medium'])
        sns.set_style(CONFIG['visualization']['style'])

        ratings = list(ratings_data['rating_counts'].keys())
        counts = list(ratings_data['rating_counts'].values())

        ax = sns.barplot(x=ratings, y=counts,
                        palette=CONFIG['visualization']['palettes']['ratings'],
                        order=ratings)

        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5), textcoords='offset points',
                fontsize=8
            )

        self._apply_plot_styling(ax, 'Content Rating Distribution',
                               'Rating', 'Content Count', rotation=45)

        plt.savefig(self.output_dir / '5_rating_distribution.png',
                   dpi=CONFIG['visualization']['dpi'], bbox_inches='tight')
        plt.close()

    def generate_all_visualizations(self, analysis_results: Dict) -> None:
        logger.info("Generating all visualizations")

        self.create_content_type_plot(analysis_results)
        self.create_countries_plot(analysis_results)
        self.create_trends_plot(analysis_results)
        self.create_genres_plot(analysis_results)
        self.create_ratings_plot(analysis_results)

        logger.info(f"All visualizations saved to {self.output_dir}")

class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_pdf_report(self, analysis_results: Dict, quality_report: Dict) -> None:
        logger.info("Generating PDF report")

        pdf_path = self.output_dir / f"{CONFIG['output']['report_filename']}.pdf"

        with PdfPages(pdf_path) as pdf:
            plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.8, 'Netflix Content Analysis Report',
                    ha='center', va='center', fontsize=24, fontweight='bold',
                    transform=plt.gca().transAxes)
            plt.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(11, 8.5))
            plt.text(0.1, 0.95, 'Data Quality Summary', fontsize=18, fontweight='bold')
            plt.text(0.1, 0.85, f'Total Records: {quality_report["total_records"]:,}', fontsize=12)
            plt.text(0.1, 0.80, f'Columns: {quality_report["columns"]}', fontsize=12)
            plt.text(0.1, 0.75, f'Duplicate Rows: {quality_report["duplicate_rows"]:,}', fontsize=12)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            plot_files = [
                '1_content_types.png',
                '2_top_countries.png',
                '3_content_trends.png',
                '4_top_genres.png',
                '5_rating_distribution.png'
            ]

            for plot_file in plot_files:
                plot_path = self.output_dir / plot_file
                if plot_path.exists():
                    img = plt.imread(plot_path)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()

        logger.info(f"PDF report saved: {pdf_path}")

    def generate_text_summary(self, analysis_results: Dict, quality_report: Dict,
                            processing_stats: Dict) -> None:
        logger.info("Generating text summary")

        summary_path = self.output_dir / f"{CONFIG['output']['summary_filename']}.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("NETFLIX CONTENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATA QUALITY ASSESSMENT\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Records Loaded: {quality_report['total_records']:,}\n")
            f.write(f"Final Records After Cleaning: {processing_stats['final_records']:,}\n")
            f.write(f"Records Removed: {processing_stats['records_removed']:,}\n")
            f.write(f"Duplicate Rows: {quality_report['duplicate_rows']:,}\n")
            f.write(f"Columns: {quality_report['columns']}\n\n")

            f.write("CONTENT ANALYSIS RESULTS\n")
            f.write("-" * 25 + "\n\n")

            ct = analysis_results['content_types']
            f.write("1. CONTENT TYPE DISTRIBUTION\n")
            f.write("-" * 32 + "\n")
            for content_type, count in ct['counts'].items():
                pct = ct['percentages'][content_type]
                f.write(f"{content_type}: {count:,} ({pct}%)\n")
            f.write("\n")

            countries = analysis_results['countries']
            f.write("2. TOP CONTENT PRODUCING COUNTRIES\n")
            f.write("-" * 35 + "\n")
            for i, (country, count) in enumerate(countries['top_countries'].items(), 1):
                f.write(f"{i}. {country}: {count:,}\n")
            f.write("\n")

            trends = analysis_results['trends']
            f.write("3. CONTENT ADDITION TRENDS\n")
            f.write("-" * 28 + "\n")
            f.write("Annual growth rates:\n")
            for content_type, rate in trends['growth_rates'].items():
                f.write(f"{content_type}: {rate:+.1f}% annually\n")
            f.write("\nRecent year activity:\n")
            yearly_data = trends['yearly_data']
            for year in sorted(yearly_data.keys())[-5:]:
                if 'Movie' in yearly_data and 'TV Show' in yearly_data:
                    movies = int(yearly_data['Movie'].get(year, 0))
                    tv_shows = int(yearly_data['TV Show'].get(year, 0))
                    f.write(f"{year}: {movies} Movies + {tv_shows} TV Shows = {movies + tv_shows} total\n")
            f.write("\n")

            genres = analysis_results['genres']
            f.write("4. TOP CONTENT GENRES\n")
            f.write("-" * 22 + "\n")
            for i, (genre, count) in enumerate(list(genres['top_genres'].items())[:10], 1):
                f.write(f"{i}. {genre}: {count:,}\n")
            f.write(f"\nTotal unique genres: {genres['total_genres']}\n\n")

            ratings = analysis_results['ratings']
            f.write("5. CONTENT RATING DISTRIBUTION\n")
            f.write("-" * 32 + "\n")
            f.write("By rating:\n")
            for rating, count in list(ratings['rating_counts'].items())[:10]:
                pct = ratings['rating_percentages'][rating]
                f.write(f"{rating}: {count:,} ({pct:.1f}%)\n")
            f.write("\nBy maturity group:\n")
            for group, count in ratings['maturity_groups'].items():
                f.write(f"{group}: {count:,}\n")
            f.write("\n")

            summary = analysis_results['summary']
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 19 + "\n")
            f.write(f"Total Content: {summary['total_titles']:,}\n")
            f.write(f"Average Titles Per Year: {summary['avg_titles_per_year']}\n")
            f.write(f"Content Date Range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}\n")
            f.write(f"Release Year Range: {summary['release_years_range']}\n\n")

            f.write("Report generated automatically by Netflix Analysis Toolkit.\n")
            f.write("For more details, refer to the accompanying PDF report.\n")

        logger.info(f"Text summary saved: {summary_path}")

def main(data_file_path: Optional[str] = None, output_dir: str = "netflix_analysis_output") -> None:
    try:
        print("=" * 60)
        print("NETFLIX ANALYSIS TOOLKIT - INDUSTRY GRADE EDITION")
        print("=" * 60)
        print()

        if not data_file_path:
            data_file_path = "/Users/sauravanand/Netflix _Analysis.py/netflix_titles.csv"

        data_path = Path(data_file_path)
        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Netflix content analysis")
        print("🔄 Loading and validating data...")

        df = DataLoader.load_csv(data_path)
        quality_report = DataLoader.validate_data_quality(df)

        print("✅ Data loaded successfully")
        print(f"   Records: {quality_report['total_records']:,}")
        print(f"   Columns: {quality_report['columns']}")
        print()

        print("🔄 Preprocessing data...")

        df_processed, processing_stats = DataPreprocessor.preprocess(df)

        print("✅ Data preprocessing complete")
        print(f"   Final records: {processing_stats['final_records']:,}")
        print(f"   Records removed: {processing_stats['records_removed']:,}")
        print()

        print("🔄 Analyzing content...")

        analysis_results = NetflixAnalyzer.perform_complete_analysis(df_processed)

        print("✅ Analysis complete")
        print(f"   Content types analyzed: {analysis_results['content_types']['unique_types']}")
        print(f"   Countries analyzed: {analysis_results['countries']['total_countries']}")
        print(f"   Genres identified: {analysis_results['genres']['total_genres']}")
        print()

        print("🔄 Generating visualizations...")

        viz_gen = VisualizationGenerator(output_path)
        viz_gen.generate_all_visualizations(analysis_results)

        print("✅ Visualizations generated")
        print(f"   Saved to: {output_path}")
        print()

        print("🔄 Generating reports...")

        report_gen = ReportGenerator(output_path)
        report_gen.generate_pdf_report(analysis_results, quality_report)
        report_gen.generate_text_summary(analysis_results, quality_report, processing_stats)

        print("✅ Reports generated")
        print(f"   PDF Report: {output_path}/{CONFIG['output']['report_filename']}.pdf")
        print(f"   Text Summary: {output_path}/{CONFIG['output']['summary_filename']}.txt")
        print()

        cleaned_data_path = output_path / "netflix_titles_cleaned.csv"
        df_processed.to_csv(cleaned_data_path, index=False)
        print(f"✅ Cleaned data saved: {cleaned_data_path}")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE - INDUSTRY GRADE!")
        print("=" * 60)
        print()
        print("📊 Key Findings:")
        content_types = analysis_results['content_types']['counts']
        print(f"   • Total content: {analysis_results['summary']['total_titles']:,}")
        print(f"   • Movies: {content_types.get('Movie', 0):,} ({analysis_results['content_types']['percentages'].get('Movie', 0):.1f}%)")
        print(f"   • TV Shows: {content_types.get('TV Show', 0):,} ({analysis_results['content_types']['percentages'].get('TV Show', 0):.1f}%)")

        top_country = list(analysis_results['countries']['top_countries'].keys())[0]
        top_count = list(analysis_results['countries']['top_countries'].values())[0]
        print(f"   • Largest producer: {top_country} ({top_count:,} titles)")

        print(f"   • Date range: {analysis_results['summary']['date_range']['earliest']} to {analysis_results['summary']['date_range']['latest']}")
        print()

    except NetflixAnalysisError as e:
        print(f"❌ ANALYSIS ERROR: {e}")
        logger.error(f"Analysis failed: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_file)
