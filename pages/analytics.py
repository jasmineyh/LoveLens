from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
	page_title="Data Insights Dashboard",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dating_app_behavior_dataset.csv"

PALETTE = {
	"pink": "#f05cc4",
	"violet": "#6f6fe9",
	"cyan": "#14b8c4",
	"gold": "#e1b11a",
	"red": "#ef4c3c",
	"dark": "#0b1020",
	"panel": "#0f1629",
	"text": "#e5e7eb",
	"muted": "#8b93a7",
}


def apply_theme() -> None:
	st.markdown(
		"""
		<style>
		.stApp {
			background:
				radial-gradient(circle at top left, rgba(240,92,196,0.14), transparent 28%),
				radial-gradient(circle at top right, rgba(111,111,233,0.12), transparent 25%),
				linear-gradient(180deg, #050816 0%, #070d19 38%, #08111e 100%);
			color: #e5e7eb;
			font-family: 'Segoe UI', 'Inter', sans-serif;
		}

		.stApp::before {
			content: "";
			position: fixed;
			inset: 0;
			pointer-events: none;
			background-image:
				linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
				linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
			background-size: 96px 96px;
			mask-image: linear-gradient(180deg, rgba(0,0,0,0.6), transparent 92%);
		}

		.block-container {
			padding-top: 1.8rem;
			padding-bottom: 2rem;
			max-width: 1500px;
		}

		[data-testid="stSidebar"] {
			background: linear-gradient(180deg, rgba(8,12,24,0.96), rgba(12,18,34,0.96));
			border-right: 1px solid rgba(255,255,255,0.05);
		}

		.hero-shell {
			position: relative;
			overflow: hidden;
			background: linear-gradient(135deg, rgba(16,24,43,0.9), rgba(8,12,24,0.92));
			border: 1px solid rgba(255,255,255,0.06);
			border-radius: 30px;
			padding: 28px 30px 26px 30px;
			box-shadow: 0 25px 70px rgba(0,0,0,0.32);
			margin-bottom: 1.4rem;
		}

		.hero-shell::after {
			content: "";
			position: absolute;
			inset: auto -80px -110px auto;
			width: 260px;
			height: 260px;
			background: radial-gradient(circle, rgba(240,92,196,0.34), transparent 70%);
			filter: blur(10px);
		}

		.eyebrow {
			display: inline-flex;
			gap: 0.45rem;
			align-items: center;
			padding: 0.42rem 0.8rem;
			border-radius: 999px;
			background: rgba(240,92,196,0.12);
			color: #ff7ed7;
			font-size: 0.92rem;
			font-weight: 700;
			letter-spacing: 0.02em;
			margin-bottom: 0.95rem;
		}

		.hero-title {
			font-size: clamp(2.5rem, 4vw, 4.6rem);
			font-weight: 900;
			letter-spacing: -0.05em;
			line-height: 0.96;
			margin: 0 0 0.8rem 0;
			color: #f8fafc;
		}

		.hero-copy {
			max-width: 980px;
			color: #aeb6c9;
			font-size: 1.04rem;
			line-height: 1.75;
			margin-bottom: 1rem;
		}

		.chip-row {
			display: flex;
			flex-wrap: wrap;
			gap: 0.65rem;
			margin-top: 0.8rem;
		}

		.feature-chip {
			display: inline-flex;
			align-items: center;
			gap: 0.45rem;
			padding: 0.55rem 0.85rem;
			border-radius: 999px;
			background: rgba(255,255,255,0.045);
			border: 1px solid rgba(255,255,255,0.06);
			color: #dce2f0;
			font-size: 0.9rem;
			font-weight: 600;
		}

		.stat-card {
			height: 100%;
			border-radius: 22px;
			padding: 1rem 1rem 0.95rem 1rem;
			border: 1px solid rgba(255,255,255,0.06);
			background: linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
			box-shadow: 0 10px 24px rgba(0,0,0,0.2);
		}

		.stat-label {
			color: #9fa7ba;
			font-size: 0.88rem;
			font-weight: 600;
			margin-bottom: 0.45rem;
		}

		.stat-value {
			color: #f8fafc;
			font-size: 2.05rem;
			font-weight: 900;
			letter-spacing: -0.04em;
			line-height: 1;
			margin-bottom: 0.35rem;
		}

		.stat-note {
			color: #98a1b5;
			font-size: 0.84rem;
		}

		.panel-title {
			font-size: 1.02rem;
			font-weight: 800;
			color: #f2f4f8;
			margin-bottom: 0.25rem;
			letter-spacing: -0.01em;
		}

		.panel-subtitle {
			color: #8992a8;
			font-size: 0.92rem;
			margin-bottom: 1rem;
			line-height: 1.6;
		}

		.bar-track {
			height: 12px;
			border-radius: 999px;
			background: rgba(255,255,255,0.06);
			overflow: hidden;
			margin-top: 0.55rem;
		}

		.bar-fill {
			height: 100%;
			border-radius: 999px;
		}

		.bar-head {
			display: flex;
			justify-content: space-between;
			align-items: baseline;
			gap: 1rem;
			color: #dbe2f1;
			font-weight: 700;
			font-size: 0.92rem;
		}

		.tab-note {
			color: #9ca5b9;
			font-size: 0.94rem;
			margin-bottom: 1rem;
		}

		.stTabs [data-baseweb="tab-list"] {
			gap: 0.35rem;
			background: rgba(8,12,22,0.72);
			border: 1px solid rgba(255,255,255,0.05);
			border-radius: 18px;
			padding: 0.25rem;
		}

		.stTabs [data-baseweb="tab"] {
			height: 2.8rem;
			border-radius: 14px;
			padding: 0 1rem;
			background: transparent;
			color: #98a1b5;
			font-weight: 600;
		}

		.stTabs [aria-selected="true"] {
			background: linear-gradient(135deg, rgba(240,92,196,0.18), rgba(111,111,233,0.2));
			color: #ffffff;
		}

		[data-testid="stMetric"] {
			background: rgba(15,22,38,0.88);
			border: 1px solid rgba(255,255,255,0.06);
			border-radius: 18px;
			padding: 0.8rem 0.95rem;
			box-shadow: 0 10px 28px rgba(0,0,0,0.18);
		}

		[data-testid="stMetricLabel"] {
			color: #9ca5b9;
			font-size: 0.84rem;
		}

		[data-testid="stMetricValue"] {
			color: #f7fafc;
			font-weight: 900;
			letter-spacing: -0.03em;
		}

		[data-testid="stDataFrame"] {
			border-radius: 18px;
			overflow: hidden;
			border: 1px solid rgba(255,255,255,0.06);
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
	df = pd.read_csv(DATA_PATH)
	df["interest_count"] = (
		df["interest_tags"]
		.fillna("")
		.astype(str)
		.str.split(",")
		.apply(lambda items: sum(1 for item in items if item.strip()))
	)
	df["is_mutual_match"] = (df["match_outcome"].astype(str).str.strip().str.lower() == "mutual match").astype(int)
	return df


@st.cache_data(show_spinner=False)
def train_model_snapshot(data: pd.DataFrame) -> dict:
	target = data["match_outcome"].astype(str)

	feature_frame = data.drop(columns=["match_outcome", "interest_tags"]).copy()
	feature_frame["interest_count"] = data["interest_count"]

	categorical_columns = feature_frame.select_dtypes(include=["object"]).columns.tolist()

	encoded = pd.get_dummies(feature_frame, columns=categorical_columns, drop_first=False)
	encoded = encoded.fillna(0)

	label_encoder = LabelEncoder()
	y = pd.Series(label_encoder.fit_transform(target), index=encoded.index)

	sample_size = min(len(encoded), 18000)
	sample_index = encoded.sample(n=sample_size, random_state=42).index
	X_sample = encoded.loc[sample_index]
	y_sample = y.loc[sample_index]

	class_counts = y_sample.value_counts()
	stratify_target = y_sample if class_counts.min() >= 2 else None

	X_train, X_test, y_train, y_test = train_test_split(
		X_sample,
		y_sample,
		test_size=0.2,
		random_state=42,
		stratify=stratify_target,
	)

	model = RandomForestClassifier(
		n_estimators=140,
		max_depth=14,
		min_samples_leaf=2,
		random_state=42,
		n_jobs=-1,
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
	grouped: dict[str, float] = {}
	for column_name, importance in feature_importances.items():
		base_name = column_name
		for categorical_name in categorical_columns:
			if column_name.startswith(f"{categorical_name}_"):
				base_name = categorical_name
				break
		grouped[base_name] = grouped.get(base_name, 0.0) + float(importance)

	feature_df = (
		pd.DataFrame({"feature": list(grouped.keys()), "importance": list(grouped.values())})
		.sort_values("importance", ascending=False)
		.reset_index(drop=True)
	)

	return {
		"accuracy": accuracy_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
		"recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
		"f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
		"features": feature_df,
		"classes": label_encoder.classes_.tolist(),
	}


def figure_style(fig: plt.Figure) -> plt.Figure:
	fig.patch.set_facecolor(PALETTE["panel"])
	for axis in fig.axes:
		axis.set_facecolor(PALETTE["panel"])
		axis.tick_params(colors="#9ca5b9")
		for spine in axis.spines.values():
			spine.set_color("#1c2740")
	return fig


def render_donut(ax: plt.Axes, values: pd.Series, colors: list[str]) -> None:
	ax.pie(
		values.values,
		startangle=95,
		colors=colors[: len(values)],
		wedgeprops=dict(width=0.38, edgecolor=PALETTE["dark"], linewidth=2),
	)
	ax.set(aspect="equal")


def render_horizontal_bars(ax: plt.Axes, series: pd.Series, color: str) -> None:
	ordered = series.sort_values(ascending=True)
	ax.barh(ordered.index, ordered.values, color=color, alpha=0.92)
	ax.grid(axis="x", color="white", alpha=0.07, linewidth=0.8)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)


def render_score_bar(label: str, value: float, color: str) -> None:
	st.markdown(
		f"""
		<div class="bar-head">
			<span>{label}</span>
			<span style="color: {color};">{value * 100:.1f}%</span>
		</div>
		<div class="bar-track">
			<div class="bar-fill" style="width: {value * 100:.1f}%; background: linear-gradient(90deg, {color}, rgba(255,255,255,0.15));"></div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_metric_cards(df: pd.DataFrame, snapshot: dict) -> None:
	unique_outcomes = df["match_outcome"].nunique()
	dominant_outcome = df["match_outcome"].value_counts().idxmax()
	dominant_share = df["match_outcome"].value_counts(normalize=True).max() * 100

	metrics = [
		("Total Records", f"{len(df):,}", "Dataset samples"),
		("Features", f"{len(df.columns)}", "Input variables"),
		("Model Accuracy", f"{snapshot['accuracy'] * 100:.1f}%", "Random Forest snapshot"),
		("Avg Swipe Ratio", f"{df['swipe_right_ratio'].mean():.2f}", "Behavioral engagement"),
		("Match Classes", f"{unique_outcomes}", "Outcome types"),
		("Top Outcome", dominant_outcome, f"{dominant_share:.1f}% of records"),
	]

	cols = st.columns(6)
	for col, (label, value, note) in zip(cols, metrics):
		with col:
			st.markdown(
				f"""
				<div class="stat-card">
					<div class="stat-label">{label}</div>
					<div class="stat-value">{value}</div>
					<div class="stat-note">{note}</div>
				</div>
				""",
				unsafe_allow_html=True,
			)


def main() -> None:
	apply_theme()
	df = load_data()
	snapshot = train_model_snapshot(df)

	highlights = [
		"EDA charts",
		"Graphs",
		"Correlation heatmap",
		"Dataset insights",
		"Feature importance",
		"Trend analysis",
	]

	st.markdown(
		"""
		<div class="hero-shell">
			<div class="eyebrow">📊 Data Insights Dashboard</div>
			<div class="hero-title">Visualization Analytics</div>
			<div class="hero-copy">
				Comprehensive exploratory data analysis and machine learning insights for the dating app behavior dataset.
				Use this panel to inspect user composition, engagement patterns, model signals, correlation structure,
				and feature influence from the same source data that powers the predictive workflow.
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	st.markdown(
		"<div class='chip-row'>" + "".join(
			f"<span class='feature-chip'>• {item}</span>" for item in highlights
		) + "</div>",
		unsafe_allow_html=True,
	)

	st.write("")
	render_metric_cards(df, snapshot)
	st.write("")

	tabs = st.tabs(["Overview", "EDA", "Trends", "Correlation", "Feature Importance", "Dataset"])

	gender_counts = df["gender"].value_counts()
	outcome_counts = df["match_outcome"].value_counts().head(8)
	education_order = list(df["education_level"].dropna().value_counts().index)
	swipe_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Late Night", "After Midnight"]

	with tabs[0]:
		st.markdown("<div class='tab-note'>A compact overview of who is using the app, what outcomes are most common, and how the model performs on a realistic training snapshot.</div>", unsafe_allow_html=True)
		col_a, col_b = st.columns([1, 1])
		with col_a:
			st.markdown("<div class='panel-title'>Gender Distribution</div><div class='panel-subtitle'>Breakdown of user gender identities</div>", unsafe_allow_html=True)
			fig, ax = plt.subplots(figsize=(7, 5))
			render_donut(
				ax,
				gender_counts,
				[PALETTE["pink"], PALETTE["violet"], PALETTE["cyan"], PALETTE["gold"], PALETTE["red"], "#1f2937"],
			)
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with col_b:
			st.markdown("<div class='panel-title'>Match Outcome Distribution</div><div class='panel-subtitle'>Results of user matching interactions</div>", unsafe_allow_html=True)
			fig, ax = plt.subplots(figsize=(7, 5))
			render_donut(
				ax,
				outcome_counts,
				[PALETTE["pink"], PALETTE["violet"], PALETTE["cyan"], PALETTE["gold"], PALETTE["red"], "#93c5fd", "#1f2937", "#7c2d12"],
			)
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)

		st.write("")
		col_c, col_d = st.columns([1, 1])
		with col_c:
			st.markdown("<div class='panel-title'>Best Model Metrics</div><div class='panel-subtitle'>Random Forest classifier performance breakdown</div>", unsafe_allow_html=True)
			render_score_bar("Accuracy", snapshot["accuracy"], PALETTE["pink"])
			st.write("")
			render_score_bar("Precision", snapshot["precision"], PALETTE["violet"])
			st.write("")
			render_score_bar("Recall", snapshot["recall"], PALETTE["cyan"])
			st.write("")
			render_score_bar("F1 Score", snapshot["f1"], PALETTE["gold"])
			st.write("")
			inner_a, inner_b = st.columns(2)
			with inner_a:
				st.metric("Overall Accuracy", f"{snapshot['accuracy'] * 100:.1f}%")
			with inner_b:
				st.metric("Macro F1", f"{snapshot['f1']:.2f}")
		with col_d:
			st.markdown("<div class='panel-title'>Dataset Snapshot</div><div class='panel-subtitle'>High-level insights extracted from the raw records</div>", unsafe_allow_html=True)
			summary_rows = [
				("Dominant gender", gender_counts.idxmax(), f"{gender_counts.max() / len(df) * 100:.1f}%"),
				("Most common outcome", outcome_counts.idxmax(), f"{outcome_counts.max() / len(df) * 100:.1f}%"),
				("Average app usage", f"{df['app_usage_time_min'].mean():.0f} min", "per user"),
				("Average messages", f"{df['message_sent_count'].mean():.1f}", "per user"),
				("Average active hour", f"{df['last_active_hour'].mean():.1f}", "24-hour clock"),
			]
			for label, value, note in summary_rows:
				st.markdown(
					f"""
					<div style="display:flex;justify-content:space-between;align-items:baseline;padding:0.55rem 0;border-bottom:1px solid rgba(255,255,255,0.06);gap:1rem;">
						<span style="color:#b2bbcd;font-weight:600;">{label}</span>
						<span style="color:#f8fafc;font-weight:800;">{value}</span>
					</div>
					<div style="color:#7f889b;font-size:0.82rem;margin-top:-0.2rem;margin-bottom:0.25rem;">{note}</div>
					""",
					unsafe_allow_html=True,
				)

	with tabs[1]:
		st.markdown("<div class='tab-note'>EDA charts focus on behavior and demographic slices to expose how engagement varies across user groups.</div>", unsafe_allow_html=True)
		left, right = st.columns(2)
		with left:
			behavior_labels = ["Optimistic", "Balanced", "Choosy", "Swipe Maniac"]
			edu_behavior = (
				df.assign(education_level=pd.Categorical(df["education_level"], categories=education_order, ordered=True))
				.groupby(["education_level", "swipe_right_label"])
				.size()
				.reset_index(name="count")
			)
			edu_behavior = edu_behavior[edu_behavior["swipe_right_label"].isin(behavior_labels)]
			pivot = (
				edu_behavior.pivot(index="education_level", columns="swipe_right_label", values="count")
				.fillna(0)
				.reindex(columns=behavior_labels)
				.reindex(education_order)
				.fillna(0)
			)
			fig, ax = plt.subplots(figsize=(8, 5))
			bottom = np.zeros(len(pivot.index))
			stack_colors = [PALETTE["pink"], PALETTE["violet"], PALETTE["cyan"], PALETTE["red"]]
			for i, column in enumerate(pivot.columns):
				values = pivot[column].values
				ax.bar(pivot.index, values, bottom=bottom, color=stack_colors[i % len(stack_colors)], label=column)
				bottom = bottom + values
			ax.set_title("Swipe Behavior by Education", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_ylabel("Record count", color="#9ca5b9")
			ax.tick_params(axis="x", rotation=32, labelcolor="#9ca5b9")
			ax.grid(axis="y", color="white", alpha=0.08)
			ax.legend(frameon=False, labelcolor="#cdd6e6", ncol=2)
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with right:
			location_summary = (
				df.groupby("location_type")["mutual_matches"]
				.agg(["count", "sum"])
				.rename(columns={"count": "users", "sum": "matches"})
				.sort_values("users", ascending=False)
			)
			fig, ax = plt.subplots(figsize=(8, 5))
			x = np.arange(len(location_summary.index))
			width = 0.35
			ax.bar(x - width / 2, location_summary["users"], width, color=PALETTE["violet"], label="users")
			ax.bar(x + width / 2, location_summary["matches"], width, color=PALETTE["pink"], label="matches")
			ax.set_xticks(x)
			ax.set_xticklabels(location_summary.index, color="#9ca5b9")
			ax.set_title("Location Type Analysis", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_ylabel("Count", color="#9ca5b9")
			ax.grid(axis="y", color="white", alpha=0.08)
			ax.legend(frameon=False, labelcolor="#cdd6e6")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)

		st.write("")
		lower_left, lower_right = st.columns(2)
		with lower_left:
			interest_tags = (
				df["interest_tags"]
				.dropna()
				.astype(str)
				.str.split(",")
				.explode()
				.str.strip()
				.replace("", np.nan)
				.dropna()
			)
			top_interests = interest_tags.value_counts().head(8)
			fig, ax = plt.subplots(figsize=(8, 5))
			render_horizontal_bars(ax, top_interests, PALETTE["cyan"])
			ax.set_title("Top Interest Tags", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_xlabel("Frequency", color="#9ca5b9")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with lower_right:
			match_time = (
				df.groupby("swipe_time_of_day")["match_outcome"]
				.count()
				.reindex(swipe_order)
				.dropna()
			)
			fig, ax = plt.subplots(figsize=(8, 5))
			ax.bar(match_time.index, match_time.values, color=PALETTE["gold"], alpha=0.9)
			ax.set_title("Match Activity by Swipe Time", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_ylabel("Records", color="#9ca5b9")
			ax.tick_params(axis="x", rotation=25, labelcolor="#9ca5b9")
			ax.grid(axis="y", color="white", alpha=0.08)
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)

	with tabs[2]:
		st.markdown("<div class='tab-note'>Trend analysis highlights how engagement intensifies during different hours and interaction phases.</div>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)
		with col1:
			hour_summary = (
				df.groupby("last_active_hour")
				.agg({"swipe_right_ratio": "mean", "mutual_matches": "mean", "message_sent_count": "mean"})
				.sort_index()
			)
			fig, ax = plt.subplots(figsize=(8.5, 5))
			ax.plot(hour_summary.index, hour_summary["swipe_right_ratio"] * 100, color=PALETTE["pink"], linewidth=2.6, marker="o", label="Swipe ratio")
			ax.plot(hour_summary.index, hour_summary["mutual_matches"], color=PALETTE["violet"], linewidth=2.6, marker="s", label="Mutual matches")
			ax.set_title("Engagement Trend by Hour", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_xlabel("Last active hour", color="#9ca5b9")
			ax.set_ylabel("Normalised values", color="#9ca5b9")
			ax.grid(color="white", alpha=0.08)
			ax.legend(frameon=False, labelcolor="#cdd6e6")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with col2:
			time_map = (
				df.groupby("swipe_time_of_day")
				.agg({"app_usage_time_min": "mean", "likes_received": "mean"})
				.reindex(swipe_order)
				.dropna()
			)
			fig, ax = plt.subplots(figsize=(8.5, 5))
			x = np.arange(len(time_map.index))
			ax.bar(x - 0.18, time_map["app_usage_time_min"], width=0.36, color=PALETTE["cyan"], label="App usage")
			ax.bar(x + 0.18, time_map["likes_received"], width=0.36, color=PALETTE["gold"], label="Likes received")
			ax.set_xticks(x)
			ax.set_xticklabels(time_map.index, color="#9ca5b9", rotation=25)
			ax.set_title("Time-of-Day Interaction Mix", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_ylabel("Average value", color="#9ca5b9")
			ax.grid(axis="y", color="white", alpha=0.08)
			ax.legend(frameon=False, labelcolor="#cdd6e6")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)

	with tabs[3]:
		st.markdown("<div class='tab-note'>Correlation explores how strongly behavior, activity, and profile richness move together.</div>", unsafe_allow_html=True)
		corr_cols = [
			"app_usage_time_min",
			"swipe_right_ratio",
			"likes_received",
			"mutual_matches",
			"profile_pics_count",
			"bio_length",
			"message_sent_count",
			"emoji_usage_rate",
			"last_active_hour",
			"interest_count",
		]
		corr = df[corr_cols].corr(numeric_only=True)
		col_left, col_right = st.columns([1.05, 0.95])
		with col_left:
			fig, ax = plt.subplots(figsize=(9, 6.7))
			sns.heatmap(
				corr,
				ax=ax,
				cmap=sns.color_palette([PALETTE["dark"], PALETTE["pink"], PALETTE["violet"], PALETTE["cyan"]], as_cmap=True),
				vmin=-1,
				vmax=1,
				center=0,
				square=True,
				linewidths=0.6,
				linecolor="#111826",
				cbar_kws={"shrink": 0.75},
			)
			ax.set_title("Feature Correlation Heatmap", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.tick_params(axis="x", labelrotation=35, colors="#9ca5b9")
			ax.tick_params(axis="y", colors="#9ca5b9")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with col_right:
			corr_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))
			strongest = corr_abs.stack().sort_values(ascending=False).head(6)
			st.markdown("<div class='panel-title'>Strongest Relationships</div><div class='panel-subtitle'>Most connected feature pairs in the dataset</div>", unsafe_allow_html=True)
			for (left, right), value in strongest.items():
				st.markdown(
					f"""
					<div style="margin-bottom:0.95rem;">
						<div class="bar-head"><span>{left} • {right}</span><span>{value:.2f}</span></div>
						<div class="bar-track"><div class="bar-fill" style="width:{value * 100:.1f}%; background: linear-gradient(90deg, {PALETTE['violet']}, {PALETTE['pink']});"></div></div>
					</div>
					""",
					unsafe_allow_html=True,
				)

	with tabs[4]:
		st.markdown("<div class='tab-note'>Feature importance is computed from a cached Random Forest trained on the raw dataset with one-hot encoded categories.</div>", unsafe_allow_html=True)
		feature_df = snapshot["features"].head(10).copy()
		col_left, col_right = st.columns([1.08, 0.92])
		with col_left:
			fig, ax = plt.subplots(figsize=(9, 6))
			render_horizontal_bars(ax, feature_df.set_index("feature")["importance"], PALETTE["pink"])
			ax.set_title("Feature Importance", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_xlabel("Importance score", color="#9ca5b9")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with col_right:
			st.markdown("<div class='panel-title'>Top Predictive Signals</div><div class='panel-subtitle'>The strongest factors shaping the match outcome classifier</div>", unsafe_allow_html=True)
			for _, row in feature_df.iterrows():
				st.markdown(
					f"""
					<div style="margin-bottom:0.9rem;">
						<div class="bar-head"><span>{row['feature']}</span><span>{row['importance'] * 100:.1f}%</span></div>
						<div class="bar-track"><div class="bar-fill" style="width:{row['importance'] * 100:.1f}%; background: linear-gradient(90deg, {PALETTE['pink']}, {PALETTE['gold']});"></div></div>
					</div>
					""",
					unsafe_allow_html=True,
				)
			st.info("The chart aggregates one-hot encoded categorical signals back to their source feature so the ranking stays readable.")

	with tabs[5]:
		st.markdown("<div class='tab-note'>Dataset insights combine schema checks, descriptive statistics, and a quick preview of the raw rows.</div>", unsafe_allow_html=True)
		col1, col2 = st.columns([1.05, 0.95])
		with col1:
			st.markdown("<div class='panel-title'>Data Quality Snapshot</div><div class='panel-subtitle'>A quick view of completeness and structure</div>", unsafe_allow_html=True)
			missing = df.isna().sum()
			quality = pd.DataFrame({
				"column": missing.index,
				"missing": missing.values,
				"dtype": df.dtypes.astype(str).values,
			}).sort_values(["missing", "column"], ascending=[False, True])
			st.dataframe(quality, use_container_width=True, height=350)
		with col2:
			st.markdown("<div class='panel-title'>Descriptive Statistics</div><div class='panel-subtitle'>Central tendency for the core behavior variables</div>", unsafe_allow_html=True)
			numeric_frame = df[[
				"app_usage_time_min",
				"swipe_right_ratio",
				"likes_received",
				"mutual_matches",
				"profile_pics_count",
				"bio_length",
				"message_sent_count",
				"emoji_usage_rate",
				"last_active_hour",
				"interest_count",
			]]
			st.dataframe(numeric_frame.describe().T, use_container_width=True, height=350)

		st.write("")
		st.markdown("<div class='panel-title'>Dataset Preview</div><div class='panel-subtitle'>A slice of the raw records used across the dashboard</div>", unsafe_allow_html=True)
		st.dataframe(df.head(12), use_container_width=True, height=330)


main()
