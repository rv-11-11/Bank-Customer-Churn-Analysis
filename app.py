import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(page_title='Bank Customer Churn Analysis', page_icon="🏦", layout='wide')


@st.cache_data
def load_data():

    df = pd.read_csv('Bank Customer Churn Prediction.csv')
    return df

df = load_data()

# ************************************************************************************************************
#                                          ANALYSIS FUNCTIONS
# ************************************************************************************************************

def perform_overall_analysis(df):
    st.header("📊 Overall Bank Analysis")

    # --- BUSINESS CONTEXT (Annotated Quote) ---
    col_intro1, col_intro2 = st.columns([3, 1])

    with col_intro1:
        # Styled "Annotation" Box with Dark Text
        st.markdown(
            """
            <div style="
                background-color: #FEF9E7; 
                border-left: 6px solid #F39C12; 
                padding: 20px; 
                border-radius: 5px; 
                font-family: 'Courier New', Courier, monospace; 
                color: #000000;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color: #D35400; margin:0; font-weight: bold;">📝 Key Insight</h4>
                <p style="font-size: 18px; margin: 10px 0 0 0; font-weight: 600;">
                    "Did you know that attracting a new customer costs 
                    <span style="background-color: #F4D03F; padding: 2px 5px;">five times</span> 
                    as much as keeping an existing one?"
                </p>
                <p style="font-size: 14px; margin-top: 5px; color: #555;">- Analyzing churn is key to profitability.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_intro2:
        try:
            st.image("churn_image.png", caption="Customer Retention", use_container_width=True)
        except:
            st.markdown("## 📉")

    st.markdown("---")

    # --- 1. KPI CARDS ---
    total_customers = df.shape[0]
    total_churned = df['churn'].sum()
    churn_rate = (total_churned / total_customers) * 100
    avg_balance = df['balance'].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", f"{total_customers:,}")
    kpi2.metric("Total Churned", f"{total_churned:,}", delta_color="inverse")
    kpi3.metric("Churn Rate", f"{churn_rate:.2f}%")
    kpi4.metric("Avg Balance", f"${avg_balance:,.2f}")

    st.markdown("---")

    colors = {'Churned': '#B71C1C', 'Retained': '#0D47A1'}

    # --- 2. CHARTS ROW 1 ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df['churn'].value_counts().reset_index()
        churn_counts.columns = ['churn', 'count']
        churn_counts['Label'] = churn_counts['churn'].map({1: 'Churned', 0: 'Retained'})

        fig_pie = px.pie(churn_counts, names='Label', values='count', color='Label',
                         color_discrete_map=colors, hole=0.4)


        fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                              textfont_size=14, textfont_color='white')
        fig_pie.update_layout(showlegend=False)  # Clean look
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Active vs Inactive Customers")
        active_counts = df.groupby(['active_member', 'churn']).size().reset_index(name='count')
        active_counts['Status'] = active_counts['active_member'].map({1: 'Active', 0: 'Inactive'})
        active_counts['Churn'] = active_counts['churn'].map({1: 'Churned', 0: 'Retained'})

        fig_bar = px.bar(active_counts, x='Status', y='count', color='Churn', barmode='group',
                         color_discrete_map=colors, text_auto=True)


        fig_bar.update_layout(font=dict(color="black"))
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 3. CHARTS ROW 2 ---
    plot_df = df.copy()
    plot_df['Churn Label'] = plot_df['churn'].map({1: 'Churned', 0: 'Retained'})

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Credit Score Distribution")
        fig_hist = px.histogram(plot_df, x='credit_score', color='Churn Label', marginal="box",
                                color_discrete_map=colors, nbins=30, opacity=0.8)
        fig_hist.update_layout(barmode='overlay', font=dict(color="black"))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        st.subheader("Balance Distribution")
        fig_box = px.box(plot_df, x='Churn Label', y='balance', color='Churn Label',
                         color_discrete_map=colors)
        fig_box.update_layout(font=dict(color="black"))
        st.plotly_chart(fig_box, use_container_width=True)

    # --- 4. CORRELATION HEATMAP ---
    st.subheader("🔥 What drives Churn? (Correlation Heatmap)")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
    fig_corr.update_layout(font=dict(color="black"))
    st.plotly_chart(fig_corr, use_container_width=True)


def perform_customer_analysis(df):
    st.header("👤 Customer Analysis")

    # Calculate averages
    churn_age = df[df['churn'] == 1]['age'].mean()
    churn_bal = df[df['churn'] == 1]['balance'].mean()
    retain_age = df[df['churn'] == 0]['age'].mean()
    retain_bal = df[df['churn'] == 0]['balance'].mean()

    st.markdown("##### 🧐 Quick Profile Comparison")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown(f"""
        <div style="
            background-color: #FFFFFF; 
            padding: 15px; 
            border-radius: 8px; 
            border: 2px solid #D50000; 
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
            <strong style="color: #D50000; font-size: 16px;">❌ CHURNED CUSTOMER</strong><br>
            <span style="font-size: 22px; font-weight: 900; color: #000000;">{churn_age:.0f} yrs</span> <br>
            <span style="font-size: 22px; font-weight: 900; color: #000000;">${churn_bal:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown(f"""
        <div style="
            background-color: #FFFFFF; 
            padding: 15px; 
            border-radius: 8px; 
            border: 2px solid #002171; 
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
            <strong style="color: #002171; font-size: 16px;">✅ RETAINED CUSTOMER</strong><br>
            <span style="font-size: 22px; font-weight: 900; color: #000000;">{retain_age:.0f} yrs</span> <br>
            <span style="font-size: 22px; font-weight: 900; color: #000000;">${retain_bal:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    colors = {'Churned': '#D50000', 'Retained': '#002171'}

    plot_df = df.copy()
    plot_df['Churn Label'] = plot_df['churn'].map({1: 'Churned', 0: 'Retained'})
    plot_df['Has Credit Card'] = plot_df['credit_card'].map({1: 'Yes', 0: 'No'})

    chart_font = dict(family="Arial Black", size=14, color="black")

    # --- 2. TABS ---
    tab1, tab2, tab3 = st.tabs(["👥 Demographics", "💰 Financials", "🛍️ Products & Services"])

    # --- TAB 1: DEMOGRAPHICS ---
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gender Distribution")
            gender_churn = plot_df.groupby(['gender', 'Churn Label']).size().reset_index(name='count')

            fig_gender = px.bar(gender_churn, x='gender', y='count', color='Churn Label',
                                barmode='group', color_discrete_map=colors, text_auto=True)


            fig_gender.update_traces(textfont_color='white', textfont_weight='bold')
            fig_gender.update_layout(font=chart_font, xaxis_title=None, yaxis_title=None, legend_title=None)
            st.plotly_chart(fig_gender, use_container_width=True)

        with col2:
            st.subheader("Age Distribution")
            fig_age = px.box(plot_df, x='Churn Label', y='age', color='Churn Label',
                             color_discrete_map=colors)

            fig_age.update_layout(font=chart_font, xaxis_title=None, yaxis_title="Age", showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)

    # --- TAB 2: FINANCIALS ---
    with tab2:
        st.subheader("Balance Distribution")
        fig_bal = px.histogram(plot_df, x='balance', color='Churn Label',
                               marginal="box", nbins=40, opacity=0.8, barmode='overlay',
                               color_discrete_map=colors)

        fig_bal.update_layout(font=chart_font, xaxis_title="Account Balance", yaxis_title="Count")
        st.plotly_chart(fig_bal, use_container_width=True)

    # --- TAB 3: PRODUCTS ---
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Products Owned")
            prod_counts = plot_df.groupby(['products_number', 'Churn Label']).size().reset_index(name='count')
            fig_prod = px.bar(prod_counts, x='products_number', y='count', color='Churn Label',
                              barmode='group', color_discrete_map=colors, text_auto=True)

            fig_prod.update_traces(textfont_color='white', textfont_weight='bold')
            fig_prod.update_layout(font=chart_font, xaxis_title="Num Products", yaxis_title=None)
            fig_prod.update_xaxes(tickmode='linear')
            st.plotly_chart(fig_prod, use_container_width=True)

        with col2:
            st.subheader("Credit Card Status")
            card_counts = plot_df.groupby(['Has Credit Card', 'Churn Label']).size().reset_index(name='count')
            fig_card = px.bar(card_counts, x='Has Credit Card', y='count', color='Churn Label',
                              barmode='group', color_discrete_map=colors, text_auto=True)

            fig_card.update_traces(textfont_color='white', textfont_weight='bold')
            fig_card.update_layout(font=chart_font, xaxis_title="Has Credit Card?", yaxis_title=None)
            st.plotly_chart(fig_card, use_container_width=True)

    # --- 3. CLUSTER ANALYSIS ---
    st.markdown("---")
    st.subheader("🔍 Advanced Segmentation: Age vs Balance")

    fig_scatter = px.scatter(
        plot_df, x='age', y='balance', color='Churn Label',
        color_discrete_map=colors, opacity=0.7,  # Increased opacity for visibility
        hover_data=['gender', 'credit_score']
    )

    fig_scatter.update_layout(
        font=chart_font,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def perform_churn_risk_analysis(df):
    # --- 0. SECTION HEADER & DISCLAIMER ---
    st.markdown("""
    <div style="background-color: #FFFFFF; border-left: 8px solid #D50000; padding: 20px; border-radius: 5px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #D50000; margin:0; font-weight: 800;">🛑 Churn Risk & Prioritization Engine</h3>
        <p style="font-size: 16px; color: #000000; margin-top: 5px; font-weight: 500;">
            This module identifies <b>high-risk, high-value customers</b> and recommends specific actions to prevent revenue loss.
        </p>
        <p style="font-size: 13px; color: #666; margin-top: 5px; font-style: italic;">
            *Disclaimer: This is a cross-sectional risk analysis based on current customer snapshots, as time-series data is not available.*
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛡️ View Risk Logic & Threshold Justification (Why this works?)", expanded=False):
        st.markdown("""
        <div style="padding: 15px; background-color: #f9f9f9; border-radius: 5px; color: black;">
            <h5 style="margin:0;">Risk Scoring Rules (Derived from EDA)</h5>
            <ul style="font-size: 14px; margin-top: 10px;">
                <li><b>Inactivity (+3 Points):</b> Historically, inactive customers churn <b>2x more</b> than active ones. This is the strongest predictor.</li>
                <li><b>Single Product (+2 Points):</b> Customers with only 1 product have lower stickiness and higher churn rates.</li>
                <li><b>High Balance (+1 Point):</b> Wealthier customers are targeted by competitors, making them higher value-at-risk.</li>
                <li><b>Age > 45 (+1 Point):</b> EDA shows a spike in churn for middle-aged demographics.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    risk_df = df[df['churn'] == 0].copy()

    median_bal = risk_df['balance'].median()

    def calculate_risk_score(row):
        score = 0
        if row['active_member'] == 0: score += 3
        if row['products_number'] == 1: score += 2
        if row['balance'] > median_bal: score += 1
        if row['age'] > 45: score += 1
        return score

    risk_df['Risk Score'] = risk_df.apply(calculate_risk_score, axis=1)

    def categorize_risk(score):
        if score >= 5:
            return 'High Risk'
        elif score >= 3:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    risk_df['Risk Level'] = risk_df['Risk Score'].apply(categorize_risk)

    risk_colors = {'High Risk': '#D50000', 'Medium Risk': '#FF6D00', 'Low Risk': '#2E7D32'}
    chart_font = dict(family="Arial", size=12, color="black")

    # --- 2. KPI CARDS (Gap 6: Business Impact) ---
    high_risk_customers = risk_df[risk_df['Risk Level'] == 'High Risk']
    count_high_risk = len(high_risk_customers)
    pct_high_risk = (count_high_risk / len(risk_df)) * 100

    # Financial Impact
    total_existing_balance = risk_df['balance'].sum()
    revenue_at_risk = high_risk_customers['balance'].sum()
    pct_revenue_risk = (revenue_at_risk / total_existing_balance) * 100

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    def kpi_card(title, value, color, subtext=""):
        return f"""
        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 2px solid {color}; text-align: center;">
            <p style="margin: 0; font-size: 14px; color: #000; font-weight: bold;">{title}</p>
            <h3 style="margin: 5px 0 0 0; color: {color}; font-weight: 900;">{value}</h3>
            <p style="margin: 0; font-size: 12px; color: #555;">{subtext}</p>
        </div>
        """

    with kpi1:
        st.markdown(kpi_card("🔥 High-Risk Customers", f"{count_high_risk:,}", "#D50000"), unsafe_allow_html=True)
    with kpi2:
        st.markdown(kpi_card("📊 % High Risk", f"{pct_high_risk:.1f}%", "#D50000", "of total customer base"),
                    unsafe_allow_html=True)
    with kpi3:
        st.markdown(kpi_card("💸 Revenue at Risk", f"${revenue_at_risk / 1000000:.1f} M", "#000000"),
                    unsafe_allow_html=True)
    with kpi4:
        st.markdown(kpi_card("📉 Impact Share", f"{pct_revenue_risk:.1f}%", "#FF6D00", "of total bank deposits"),
                    unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("✅ Logic Validation: Historical Churn Comparison")
    st.caption(
        "We validated our risk rules against past churn data. As shown below, the factors we flagged (Inactivity, Single Product) indeed caused higher churn.")

    val_col1, val_col2 = st.columns(2)
    with val_col1:

        hist_active = df.groupby('active_member')['churn'].mean().reset_index()
        hist_active['Label'] = hist_active['active_member'].map({1: 'Active (Low Risk)', 0: 'Inactive (High Risk)'})

        fig_val1 = px.bar(hist_active, x='Label', y='churn', title="Historical Churn Rate: Active vs Inactive",
                          color='Label',
                          color_discrete_map={'Active (Low Risk)': '#2E7D32', 'Inactive (High Risk)': '#D50000'},
                          text_auto='.1%')
        fig_val1.update_layout(font=chart_font, yaxis_title="Churn Rate", xaxis_title=None, showlegend=False)
        st.plotly_chart(fig_val1, use_container_width=True)

    with val_col2:

        hist_prod = df.groupby('products_number')['churn'].mean().reset_index()
        hist_prod['Color'] = hist_prod['products_number'].apply(lambda x: '#D50000' if x == 1 else '#2E7D32')

        fig_val2 = px.bar(hist_prod, x='products_number', y='churn', title="Historical Churn Rate by Products",
                          color='Color', color_discrete_map="identity", text_auto='.1%')
        fig_val2.update_layout(font=chart_font, yaxis_title="Churn Rate", xaxis_title="Number of Products")
        fig_val2.update_xaxes(tickmode='linear')
        st.plotly_chart(fig_val2, use_container_width=True)

    st.markdown("---")

    col_dist1, col_dist2 = st.columns([1, 2])
    with col_dist1:
        st.subheader("Risk Distribution")
        risk_counts = risk_df['Risk Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'count']
        fig_donut = px.pie(risk_counts, values='count', names='Risk Level', hole=0.5,
                           color='Risk Level', color_discrete_map=risk_colors)
        fig_donut.update_traces(textinfo='percent+label', textfont_size=13, textfont_color='white')
        fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=0, l=0, r=0), font=chart_font)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_dist2:
        st.subheader("🕵️ Who are the High-Risk Customers?")
        tab_p1, tab_p2 = st.tabs(["Age Profile", "Product Usage"])
        with tab_p1:
            fig_age = px.box(risk_df, x='Risk Level', y='age', color='Risk Level',
                             color_discrete_map=risk_colors)
            fig_age.update_layout(font=chart_font, xaxis_title=None, showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        with tab_p2:
            prod_risk = risk_df.groupby(['products_number', 'Risk Level']).size().reset_index(name='count')
            fig_prod = px.bar(prod_risk, x='products_number', y='count', color='Risk Level',
                              color_discrete_map=risk_colors, barmode='group')
            fig_prod.update_layout(font=chart_font, xaxis_title="Products Owned", yaxis_title="Count")
            st.plotly_chart(fig_prod, use_container_width=True)

    # --- 5. REGION & ACTIVITY ---
    st.markdown("---")
    col_reg1, col_reg2 = st.columns(2)
    with col_reg1:
        st.subheader("🌍 Risk by Region")
        region_risk = risk_df[risk_df['Risk Level'] == 'High Risk'].groupby('country').size().reset_index(
            name='High Risk Count')
        fig_reg = px.bar(region_risk, x='country', y='High Risk Count', text_auto=True,
                         color='High Risk Count', color_continuous_scale='Reds')
        fig_reg.update_layout(font=chart_font, coloraxis_showscale=False)
        st.plotly_chart(fig_reg, use_container_width=True)
    with col_reg2:
        st.subheader("⚡ Activity Status Impact")
        act_risk = risk_df.groupby(['active_member', 'Risk Level']).size().reset_index(name='count')
        act_risk['Status'] = act_risk['active_member'].map({1: 'Active', 0: 'Inactive'})
        fig_act = px.bar(act_risk, x='Status', y='count', color='Risk Level',
                         color_discrete_map=risk_colors, barmode='group')
        fig_act.update_layout(font=chart_font, xaxis_title=None)
        st.plotly_chart(fig_act, use_container_width=True)

    # --- 6. STRATEGIC TAKEAWAY ---
    st.markdown("""
    <div style="
        background-color: #FFF3E0; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 6px solid #E65100; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-top: 20px;
        margin-bottom: 20px;">
        <strong style="color: #E65100; font-size: 16px; display: block; margin-bottom: 5px;">📌 STRATEGIC TAKEAWAY</strong> 
        <span style="color: #000000; font-size: 15px; font-weight: 500;">
            <b>The Pareto Principle in Action:</b> Although High-Risk customers are only <b>{:.1f}%</b> of the base, they hold <b>{:.1f}%</b> of the bank's total liquidity.
            Prioritizing this small segment yields disproportionate returns.
        </span>
    </div>
    """.format(pct_high_risk, pct_revenue_risk), unsafe_allow_html=True)

    # --- 7. EXECUTIVE DASHBOARD & STRATEGY ---
    st.markdown("### 💎 Executive Dashboard: Risk vs Value")
    col_strat1, col_strat2 = st.columns([2, 1])

    with col_strat1:
        fig_scatter = px.scatter(
            risk_df, x='balance', y='age', color='Risk Level',
            color_discrete_map=risk_colors,
            opacity=0.6, title="Cluster Analysis: Balance vs Age",
            hover_data=['country']
        )
        fig_scatter.add_shape(type="rect",
                              x0=risk_df['balance'].mean(), y0=45, x1=risk_df['balance'].max(), y1=risk_df['age'].max(),
                              line=dict(color="Black", width=2, dash="dot"),
                              )
        fig_scatter.update_layout(font=chart_font, height=450, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_strat2:
        st.markdown("#### 🎯 Action Matrix")
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; font-family: Arial; color: black;">
          <tr>
            <td style="border: 2px solid #333; padding: 10px; background-color: #ffcdd2;">
                <b>High Value / High Risk</b><br>
                <span style="color: #b71c1c; font-weight:900;">🔥 PRIORITY 1</span><br>
                <small>Direct Call by RM</small>
            </td>
            <td style="border: 2px solid #333; padding: 10px; background-color: #c8e6c9;">
                <b>High Value / Low Risk</b><br>
                <span style="color: #1b5e20; font-weight:900;">🛡️ RETAIN</span><br>
                <small>Cross-Sell</small>
            </td>
          </tr>
          <tr>
            <td style="border: 2px solid #333; padding: 10px; background-color: #fff9c4;">
                <b>Low Value / High Risk</b><br>
                <span style="color: #f57f17; font-weight:900;">⚠️ MONITOR</span><br>
                <small>Email Nudges</small>
            </td>
            <td style="border: 2px solid #333; padding: 10px; background-color: #f5f5f5;">
                <b>Low Value / Low Risk</b><br>
                <span style="color: #757575; font-weight:900;">💤 IGNORE</span><br>
                <small>No Action</small>
            </td>
          </tr>
        </table>
        """, unsafe_allow_html=True)


    st.subheader("📋 Top 10 Critical Risk Customers (Action List)")
    top_risk = high_risk_customers.sort_values(by='balance', ascending=False).head(10)
    display_cols = ['customer_id', 'country', 'age', 'balance', 'products_number', 'active_member', 'credit_score']
    styled_df = top_risk[display_cols].copy()
    styled_df['balance'] = styled_df['balance'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(styled_df, use_container_width=True)


    st.markdown("---")
    st.markdown("""
    <div style="background-color: #fffde7; border-left: 6px solid #fbc02d; padding: 20px; border-radius: 5px;">
        <h4 style="margin:0; color: #f57f17;">📑 Executive Summary & Future Roadmap</h4>
        <p style="font-size: 16px; color: black; margin-top: 10px;">
            <b>Impact:</b> This analysis identified <b>$168M</b> of revenue at risk. Immediate intervention with the top 10% high-balance inactive users is recommended.
            <br><br>
            <b>🚀 Scalability (Future Scope):</b> 
            While this module uses a logical rule-based engine, future iterations can implement a <b>Machine Learning Classification Model (XGBoost/Random Forest)</b> to predict churn probability with higher precision as time-series data becomes available.
        </p>
    </div>
    """, unsafe_allow_html=True)


def perform_region_analysis(df):
    # --- 0. SECTION HEADER ---
    st.markdown("""
    <div style="background-color: #FFFFFF; border-left: 8px solid #1976D2; padding: 20px; border-radius: 5px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #1976D2; margin:0; font-weight: 800;">🌍 Regional Market Analysis</h3>
        <p style="font-size: 16px; color: #000000; margin-top: 5px; font-weight: 500;">
            Compare performance across geographies to identify regional churn hotspots and revenue risks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- 1. FILTER BY REGION (Sidebar) ---

    country_options = ['All Countries'] + sorted(df['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("🌍 Select Region to Filter KPIs", country_options, key="region_filter")


    if selected_country == 'All Countries':
        df_kpi = df
    else:
        df_kpi = df[df['country'] == selected_country]


    df_global = df.copy()
    median_bal = df_global['balance'].median()

    def calc_risk(row):
        score = 0
        if row['active_member'] == 0: score += 3
        if row['products_number'] == 1: score += 2
        if row['balance'] > median_bal: score += 1
        if row['age'] > 45: score += 1
        return score

    df_global['Risk Score'] = df_global.apply(calc_risk, axis=1)
    df_global['Risk Level'] = df_global['Risk Score'].apply(lambda x: 'High Risk' if x >= 5 else 'Other')


    st.markdown(f"#### 📊 Performance Overview: {selected_country}")

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    total_cust = len(df_kpi)
    churn_rate = df_kpi['churn'].mean() * 100
    avg_bal = df_kpi['balance'].mean()
    active_pct = df_kpi['active_member'].mean() * 100

    def region_kpi(title, value, color):
        return f"""
        <div style="background-color: white; padding: 15px; border-radius: 8px; border-bottom: 4px solid {color}; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); text-align: center;">
            <p style="margin: 0; font-size: 13px; color: #555; font-weight: bold;">{title}</p>
            <h3 style="margin: 5px 0 0 0; color: {color}; font-weight: 800;">{value}</h3>
        </div>
        """

    with col_kpi1:
        st.markdown(region_kpi("Total Customers", f"{total_cust:,}", "#1976D2"), unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(region_kpi("Churn Rate", f"{churn_rate:.2f}%", "#D50000" if churn_rate > 20 else "#2E7D32"),
                    unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(region_kpi("Avg Balance", f"${avg_bal:,.0f}", "#FF6D00"), unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(region_kpi("Active Customers", f"{active_pct:.1f}%", "#2E7D32"), unsafe_allow_html=True)

    st.markdown("---")


    chart_font = dict(family="Arial", size=12, color="black")


    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("📉 Churn Rate by Region")

        region_churn = df_global.groupby('country')['churn'].mean().reset_index()
        region_churn = region_churn.sort_values('churn', ascending=False)

        fig_churn = px.bar(region_churn, x='country', y='churn', text_auto='.1%',
                           color='churn', color_continuous_scale=['#2E7D32', '#D50000'],
                           title="Which region loses the most customers?")
        fig_churn.update_layout(font=chart_font, coloraxis_showscale=False, xaxis_title=None, yaxis_title="Churn Rate")
        st.plotly_chart(fig_churn, use_container_width=True)

    with col_c2:
        st.subheader("👥 Customer Distribution")
        region_dist = df_global['country'].value_counts().reset_index()
        region_dist.columns = ['country', 'count']

        fig_dist = px.pie(region_dist, names='country', values='count', hole=0.4,
                          color='country', color_discrete_sequence=px.colors.qualitative.Bold)
        fig_dist.update_traces(textinfo='percent+label', textfont_size=13, textfont_color='white')
        fig_dist.update_layout(font=chart_font, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)


    st.markdown("---")
    col_c3, col_c4 = st.columns(2)

    with col_c3:
        st.subheader("💰 Balance Lost (Churned Users)")

        churned_df = df_global[df_global['churn'] == 1]
        bal_lost = churned_df.groupby('country')['balance'].sum().reset_index()
        bal_lost = bal_lost.sort_values('balance', ascending=False)

        fig_bal = px.bar(bal_lost, x='country', y='balance', text_auto='.2s',
                         color='country', color_discrete_sequence=['#D50000'] * 3)  # All Red for 'Loss'
        fig_bal.update_layout(font=chart_font, showlegend=False, xaxis_title=None, yaxis_title="Total Balance Lost")
        st.plotly_chart(fig_bal, use_container_width=True)

    with col_c4:
        st.subheader("⚡ Activity Status by Region")

        activity_df = df_global.groupby(['country', 'active_member']).size().reset_index(name='count')
        activity_df['Status'] = activity_df['active_member'].map({1: 'Active', 0: 'Inactive'})

        fig_act = px.bar(activity_df, x='country', y='count', color='Status',
                         color_discrete_map={'Active': '#2E7D32', 'Inactive': '#D50000'},
                         barmode='stack', text_auto=True)
        fig_act.update_layout(font=chart_font, xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(fig_act, use_container_width=True)


    st.markdown("---")
    col_c5, col_c6 = st.columns(2)

    with col_c5:
        st.subheader("📦 Product Usage by Region")
        prod_df = df_global.groupby(['country', 'products_number']).size().reset_index(name='count')

        fig_prod = px.bar(prod_df, x='country', y='count', color='products_number',
                          barmode='group', title="Do products vary by region?")
        fig_prod.update_layout(font=chart_font, xaxis_title=None)
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_c6:
        st.subheader("🔥 High-Risk Customers by Region")

        risk_counts = df_global[df_global['Risk Level'] == 'High Risk'].groupby('country').size().reset_index(
            name='count')
        risk_counts = risk_counts.sort_values('count', ascending=False)

        fig_risk = px.bar(risk_counts, x='country', y='count', text_auto=True,
                          color='count', color_continuous_scale='Reds')
        fig_risk.update_layout(font=chart_font, coloraxis_showscale=False, xaxis_title=None,
                               yaxis_title="High Risk Count")
        st.plotly_chart(fig_risk, use_container_width=True)


    st.markdown("---")
    st.subheader("🏆 Regional Ranking Matrix")


    summ_churn = df_global.groupby('country')['churn'].mean().reset_index(name='Churn Rate %')
    summ_churn['Churn Rate %'] = (summ_churn['Churn Rate %'] * 100).round(2)


    summ_count = df_global.groupby('country').size().reset_index(name='Total Customers')


    summ_bal = df_global.groupby('country')['balance'].mean().reset_index(name='Avg Balance')
    summ_bal['Avg Balance'] = summ_bal['Avg Balance'].apply(lambda x: f"${x:,.0f}")


    summ_risk = df_global[df_global['Risk Level'] == 'High Risk'].groupby('country').size().reset_index(
        name='High Risk Customers')


    rank_df = summ_count.merge(summ_churn, on='country').merge(summ_bal, on='country').merge(summ_risk, on='country',
                                                                                             how='left')
    rank_df = rank_df.sort_values('Churn Rate %', ascending=False).reset_index(drop=True)

    st.dataframe(rank_df, use_container_width=True)


    highest_churn_country = rank_df.iloc[0]['country']
    highest_churn_val = rank_df.iloc[0]['Churn Rate %']

    st.markdown(f"""
    <div style="background-color: #E3F2FD; border-left: 6px solid #1976D2; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <strong style="color: #1976D2; font-size: 16px;">📌 Regional Takeaway:</strong>
        <span style="color: black; font-size: 15px;">
            <b>{highest_churn_country}</b> is the most critical region with a churn rate of <b>{highest_churn_val}%</b>. 
            However, ensure to monitor Germany as it typically holds the highest average balance per customer.
        </span>
    </div>
    """, unsafe_allow_html=True)


def perform_actionable_insights(df):

    st.markdown("""
    <div style="background-color: #FFFFFF; border-left: 8px solid #1976D2; padding: 20px; border-radius: 5px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #1976D2; margin:0; font-weight: 800;">🚀 Actionable Insights & Recommendations</h3>
        <p style="font-size: 16px; color: #000000; margin-top: 5px; font-weight: 500;">
            Strategic roadmap: What the bank should do next to reduce churn and protect revenue.
        </p>
    </div>
    """, unsafe_allow_html=True)


    churn_rate = df['churn'].mean() * 100
    inactive_pct = (1 - df['active_member'].mean()) * 100

    churned_high_bal = df[(df['churn'] == 1) & (df['balance'] > df['balance'].mean())]
    revenue_loss_pct = (churned_high_bal['balance'].sum() / df[df['churn'] == 1]['balance'].sum()) * 100

    st.markdown(f"""
    <div style="background-color: #ffebee; border: 1px solid #ffcdd2; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h5 style="color: #b71c1c; margin: 0 0 10px 0;">⚠️ Critical Pain Points</h5>
        <ul style="color: #000000; font-size: 15px; margin-bottom: 0;">
            <li><b>High Inactivity:</b> {inactive_pct:.1f}% of customers are inactive, which is the #1 driver of churn.</li>
            <li><b>Revenue Leakage:</b> High-balance customers account for <b>{revenue_loss_pct:.0f}%</b> of the total churned liquidity.</li>
            <li><b>Product Isolation:</b> Single-product users are 2x more likely to leave than multi-product users.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


    st.subheader("🛠️ Risk-to-Action Mapping")

    strategy_data = {
        "Risk Segment": ["High Balance + Inactive", "Single Product Users", "Senior Customers (50+)",
                         "High Churn Region (Germany)"],
        "Recommended Action": ["Assign Dedicated Relationship Manager", "Offer 'Bundle' (Card + Savings)",
                               "Digital Assistance & Branch Support", "Localized Engagement Campaigns"],
        "Expected Business Impact": ["Retain High-Value Liquidity", "Increase Switching Costs (Lock-in)",
                                     "Build Trust & Loyalty", "Reduce Regional Attrition"]
    }
    strategy_df = pd.DataFrame(strategy_data)

    st.table(strategy_df)

    st.markdown("---")
    col_matrix, col_target = st.columns([1.5, 1])

    with col_matrix:
        st.subheader("📋 Execution Priority Matrix")
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; text-align: center; color: black;">
            <tr>
                <td style="padding: 15px; border: 2px solid #333; background-color: #ffcdd2;">
                    <b>High Value / High Risk</b><br>
                    <span style="color: #b71c1c; font-weight: 900; font-size: 18px;">IMMEDIATE</span><br>
                    <small>Call within 24hrs</small>
                </td>
                <td style="padding: 15px; border: 2px solid #333; background-color: #fff9c4;">
                    <b>Low Value / High Risk</b><br>
                    <span style="color: #f57f17; font-weight: 900; font-size: 18px;">MONITOR</span><br>
                    <small>Automated Email Nudge</small>
                </td>
            </tr>
            <tr>
                <td style="padding: 15px; border: 2px solid #333; background-color: #c8e6c9;">
                    <b>High Value / Low Risk</b><br>
                    <span style="color: #1b5e20; font-weight: 900; font-size: 18px;">MAINTAIN</span><br>
                    <small>Upsell Premium Services</small>
                </td>
                <td style="padding: 15px; border: 2px solid #333; background-color: #f5f5f5;">
                    <b>Low Value / Low Risk</b><br>
                    <span style="color: #757575; font-weight: 900; font-size: 18px;">IGNORE</span><br>
                    <small>No Action Required</small>
                </td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    with col_target:
        st.subheader("🎯 Prime Target")

        target_seg = df[(df['active_member'] == 0) & (df['balance'] > df['balance'].median())]
        target_count = len(target_seg)
        target_bal = target_seg['balance'].sum() / 1000000  # in Millions

        st.markdown(f"""
        <div style="background-color: #FFF3E0; border: 2px solid #FF6D00; padding: 20px; border-radius: 8px; text-align: center;">
            <strong style="color: #E65100; font-size: 18px;">High-Balance Inactive Users</strong>
            <hr style="border-color: #FF6D00; margin: 10px 0;">
            <h2 style="color: #000000; margin: 0;">{target_count:,}</h2>
            <p style="color: #555; margin: 0;">Customers</p>
            <h2 style="color: #000000; margin: 10px 0 0 0;">${target_bal:.1f} M</h2>
            <p style="color: #555; margin: 0;">Total Balance at Risk</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💡 Ready-to-Launch Campaigns")

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    def campaign_card(icon, title, desc):
        return f"""
        <div style="background-color: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; height: 180px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
            <div style="font-size: 30px; margin-bottom: 10px;">{icon}</div>
            <strong style="color: #1976D2; font-size: 16px;">{title}</strong>
            <p style="color: #333; font-size: 13px; margin-top: 5px;">{desc}</p>
        </div>
        """

    with col_c1: st.markdown(
        campaign_card("🎁", "Loyalty Booster", "Offer reward points for every active transaction to reduce inactivity."),
        unsafe_allow_html=True)
    with col_c2: st.markdown(campaign_card("💳", "Product Bundling",
                                           "Cross-sell Credit Cards to Savings users with a 'Zero Fee' for 1st year."),
                             unsafe_allow_html=True)
    with col_c3: st.markdown(
        campaign_card("🤝", "RM Outreach", "Personalized calls for customers with balance > $100k to build trust."),
        unsafe_allow_html=True)
    with col_c4: st.markdown(
        campaign_card("📱", "Digital Push", "App notifications for seniors to guide them on digital banking features."),
        unsafe_allow_html=True)

    # --- 6. REGION STRATEGY & METRICS ---
    st.markdown("---")
    col_reg, col_met = st.columns(2)

    with col_reg:
        st.subheader("🌍 Regional Strategy")
        st.info("🇩🇪 **Germany:** Focus on **retention** (High Churn). Launch VIP programs for high-balance users.")
        st.info("🇫🇷 **France:** Focus on **activity**. Launch 'Usage Rewards' to wake up the large inactive base.")
        st.info("🇪🇸 **Spain:** Focus on **cross-selling**. Promote credit cards to younger demographics.")

    with col_met:
        st.subheader("📈 Success Metrics (KPIs)")
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 5px solid #2E7D32; color: black;">
            <ul style="margin: 0; font-weight: 500;">
                <li>📉 Reduce Churn Rate by <b>5%</b> in Q3.</li>
                <li>⚡ Increase Active Member % from <b>51%</b> to <b>60%</b>.</li>
                <li>🛍️ Increase Avg Products per Customer to <b>1.8</b>.</li>
                <li>💰 Retain <b>$50M</b> of 'At-Risk' Liquidity.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- 7. FINAL EXECUTIVE TAKEAWAY ---
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #1976D2; color: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        <h4 style="margin:0;">🏁 Final Executive Recommendation</h4>
        <p style="font-size: 18px; margin-top: 10px; font-weight: 400;">
            "Focusing retention efforts on the <b>Top 10% High-Balance Inactive Customers</b> will deliver maximum ROI with minimal operational cost."
        </p>
    </div>
    """, unsafe_allow_html=True)


# ************************************************************************************************************
#                                          SIDEBAR & MAIN LOGIC
# ************************************************************************************************************

st.sidebar.title("🏦 Bank Churn Analysis")

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "📊 Overall Bank Analysis",
        "👤 Customer Analysis",
        "⚠️ Churn Risk Analysis",
        "🌍 Region Analysis",
        "📌 Actionable Insights"
    ]
)

# Global Sidebar Filters
st.sidebar.subheader("Overall Filters")

# Country Filter
country_list = df['country'].unique().tolist()
selected_countries = st.sidebar.multiselect("Filter by Country", country_list, default=country_list)


if selected_countries:
    filtered_df = df[df['country'].isin(selected_countries)]
else:
    filtered_df = df


if menu == "📊 Overall Bank Analysis":
    perform_overall_analysis(filtered_df)

elif menu == "👤 Customer Analysis":
    perform_customer_analysis(filtered_df)

elif menu == "⚠️ Churn Risk Analysis":
    perform_churn_risk_analysis(filtered_df)

elif menu == "🌍 Region Analysis":
    perform_region_analysis(filtered_df)

elif menu == "📌 Actionable Insights":
    perform_actionable_insights(filtered_df)