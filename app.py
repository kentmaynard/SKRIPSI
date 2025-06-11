import streamlit as st
import pandas as pd
import mysql.connector
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi koneksi database
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "skripsi2"
}

# Fungsi koneksi database
def connect_db():
    return mysql.connector.connect(**db_config)

# Fungsi untuk menentukan nama tabel berdasarkan platform dan jenis tabel
def get_table_name(base_table, platform=None, is_item_table=False):
    if platform is None:
        platform = st.session_state.get("bundling_type", "General").lower()
    else:
        platform = platform.lower()
    if base_table in ["deadstock", "category"]:
        return base_table
    else:
        table_name = f"{base_table}_{platform}" if platform != "general" else base_table
        if is_item_table:
            table_name += "_item"
        return table_name

# Fungsi untuk memastikan skema tabel item benar
def ensure_correct_item_table_schema():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        platforms = ["general", "shopee", "tokopedia", "lazada"]
        for platform in platforms:
            for base_table in ["bundling_all", "bundling_similarity", "bundling_ds"]:
                table_name = get_table_name(base_table, platform, is_item_table=True)
                # Cek skema tabel saat ini
                cursor.execute(f"SHOW KEYS FROM {table_name} WHERE Key_name = 'PRIMARY'")
                primary_keys = cursor.fetchall()
                current_pk = [key[4] for key in primary_keys]  # Kolom ke-4 adalah nama kolom
                expected_pk = ['id_bundling', 'id_barang', 'role']
                
                if sorted(current_pk) != sorted(expected_pk):
                    # Drop primary key yang ada
                    cursor.execute(f"ALTER TABLE {table_name} DROP PRIMARY KEY")
                    # Tambahkan primary key baru
                    cursor.execute(f"ALTER TABLE {table_name} ADD PRIMARY KEY (id_bundling, id_barang, role)")
                    st.write(f"[DEBUG] Skema tabel {table_name} diperbarui dengan PRIMARY KEY (id_bundling, id_barang, role)")
        conn.commit()
        st.success("✅ Skema tabel item berhasil diperiksa dan diperbarui jika perlu.")
    except mysql.connector.Error as e:
        st.error(f"❌ Gagal memperbarui skema tabel item: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def check_platform_data(platform):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        table_name = get_table_name('bundling_all', platform)
        query = f"SELECT COUNT(*) FROM {table_name}"
        st.write(f"[DEBUG] Query yang dijalankan: {query}")
        cursor.execute(query)
        count = cursor.fetchone()[0]
        st.write(f"[DEBUG] Hasil COUNT: {count}")
        return count > 0
    except mysql.connector.Error as e:
        st.error(f"[ERROR] Error database saat cek {table_name}: {str(e)}")
        return False
    except Exception as e:
        st.error(f"[ERROR] Error tak terduga saat cek {table_name}: {str(e)}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# Fungsi untuk memeriksa apakah tabel memiliki data
def check_tables_have_data():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {get_table_name('bundling_all')}")
        count_bundling_all = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM {get_table_name('bundling_similarity')}")
        count_bundling_similarity = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM {get_table_name('bundling_ds')}")
        count_bundling_ds = cursor.fetchone()[0]
        return count_bundling_all > 0 or count_bundling_similarity > 0 or count_bundling_ds > 0
    except Exception as e:
        st.error(f"❌ Gagal memeriksa database: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

# Fungsi untuk mendapatkan parameter FP-Growth berdasarkan tipe bundling
def get_fp_growth_params(bundling_type):
    params = {
        "General": {"minSupport": 0.001, "minConfidence": 0.3},
        "Shopee": {"minSupport": 0.005, "minConfidence": 0.1},
        "Tokopedia": {"minSupport": 0.003, "minConfidence": 0.4},
        "Lazada": {"minSupport": 0.01, "minConfidence": 0.2}
    }
    return params.get(bundling_type, {"minSupport": 0.005, "minConfidence": 0.5})

def get_fp_growth_ds_params(bundling_type):
    params = {
        "General": {"minSupport": 0.0005, "minConfidence": 0.5},
        "Shopee": {"minSupport": 0.005, "minConfidence": 0.5},
        "Tokopedia": {"minSupport": 0.005, "minConfidence": 0.5},
        "Lazada": {"minSupport": 0.03, "minConfidence": 0.5}
    }
    return params.get(bundling_type, {"minSupport": 0.005, "minConfidence": 0.5})

# Fungsi scoring RFM
def score_recency(days, years):
    if days == 0:
        return 5
    elif 1*years <= days <= 7*years:
        return 4
    elif 8*years <= days <= 30*years:
        return 3
    elif 31*years <= days <= 90*years:
        return 2
    else:
        return 1

def score_frequency(freq, years):
    freq_per_year = freq / years
    if freq_per_year >= 1200:
        return 5
    elif freq_per_year >= 600:
        return 4
    elif freq_per_year >= 300:
        return 3
    elif freq_per_year >= 120:
        return 2
    else:
        return 1

def score_monetary(amount, years):
    amount_per_year = amount / years
    if amount_per_year >= 6000:
        return 5
    elif amount_per_year >= 3000:
        return 4
    elif amount_per_year >= 1200:
        return 3
    elif amount_per_year >= 120:
        return 2
    else:
        return 1

def categorize_product(row):
    R, F, M = row["R_Score"], row["F_Score"], row["M_Score"]
    if R > 3 and F > 4:
        return "Barang Naik Daun (Trending Products)"
    elif R > 3 and F > 3:
        return "Barang Fast Moving"
    elif R > 4 and F < 3:
        return "Barang Loyal (Loyal Products)"
    elif R > 4 and 3 < F < 5:
        return "Barang Potensial (Potential Products)"
    elif 1 < F < 5 and M < 4:
        return "Barang Sedang (Average Products)"
    elif R > 2 and F <= 2:
        return "Barang Stagnan (Stagnant Products)"
    elif F < 2 and M < 2:
        return "Deadstock (Non-Moving Products)"
    elif M > 1:
        return "Barang Habis Pakai"
    else:
        return "Uncategorized"

def process_rfm_from_db(tahun_tren):
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT p.id_barang, b.deskripsi_brg, p.tanggal, p.kode_nota, p.jml
            FROM {get_table_name('penjualan')} p
            JOIN {get_table_name('barang')} b ON p.id_barang = b.id_barang
        """)
        data = cursor.fetchall()

        df = pd.DataFrame(data, columns=["id_barang", "deskripsi_brg", "tanggal", "kode_nota", "jml"])
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        reference_date = df["tanggal"].max()
        cutoff_date = reference_date - pd.DateOffset(years=tahun_tren)
        df_filtered = df[df["tanggal"] >= cutoff_date]

        rfm = df_filtered.groupby("deskripsi_brg").agg(
            Recency=("tanggal", lambda x: (reference_date - x.max()).days),
            Frequency=("kode_nota", "nunique"),
            Monetary=("jml", "sum")
        ).reset_index()

        rfm["R_Score"] = rfm["Recency"].apply(score_recency, args=(1,))
        rfm["F_Score"] = rfm["Frequency"].apply(score_frequency, args=(1,))
        rfm["M_Score"] = rfm["Monetary"].apply(score_monetary, args=(1,))
        rfm["Category"] = rfm.apply(categorize_product, axis=1)

        for _, row in rfm.iterrows():
            cursor.execute(f"SELECT id_category FROM {get_table_name('category')} WHERE category = %s", (row["Category"],))
            category_result = cursor.fetchone()
            if category_result:
                id_category = category_result[0]
            else:
                cursor.execute(f"INSERT INTO {get_table_name('category')} (category) VALUES (%s)", (row["Category"],))
                id_category = cursor.lastrowid

            cursor.execute(f"""
                INSERT INTO {get_table_name('rfm')} (recency, frequency, monetary, r_score, f_score, m_score, id_barang, id_category)
                VALUES (%s, %s, %s, %s, %s, %s, 
                        (SELECT id_barang FROM {get_table_name('barang')} WHERE deskripsi_brg = %s), %s)
            """, (
                row["Recency"], row["Frequency"], row["Monetary"],
                row["R_Score"], row["F_Score"], row["M_Score"],
                row["deskripsi_brg"], id_category
            ))

        conn.commit()
        st.success("✅ RFM dan kategori berhasil dihitung dan disimpan ke database.")
    except Exception as e:
        st.error(f"❌ Error in process_rfm_from_db: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def run_fp_growth_and_store_to_db():
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT p.kode_nota, b.deskripsi_brg
            FROM {get_table_name('penjualan')} p
            JOIN {get_table_name('barang')} b ON p.id_barang = b.id_barang
        """)
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=["kode_nota", "deskripsi_brg"])

        nota_counts = df.groupby("kode_nota")["deskripsi_brg"].count().reset_index(name="count")
        valid_nota = nota_counts[nota_counts["count"] > 1]["kode_nota"]
        df_filtered = df[df["kode_nota"].isin(valid_nota)]

        df_grouped = df_filtered.groupby("kode_nota")["deskripsi_brg"].apply(set).reset_index()
        df_grouped["items"] = df_grouped["deskripsi_brg"].apply(list)

        spark = SparkSession.builder.appName("FP_Growth_DB").getOrCreate()
        df_spark = spark.createDataFrame(df_grouped[["items"]])

        params = get_fp_growth_params(st.session_state.bundling_type)
        fp_growth = FPGrowth(itemsCol="items", minSupport=params["minSupport"], minConfidence=params["minConfidence"])
        model = fp_growth.fit(df_spark)

        rules_df = model.associationRules.toPandas()

        cursor.execute(f"DELETE FROM {get_table_name('bundling_all')}")
        cursor.execute(f"DELETE FROM {get_table_name('bundling_all', is_item_table=True)}")
        cursor.execute(f"ALTER TABLE {get_table_name('bundling_all')} AUTO_INCREMENT = 1")

        seen_rules = set()
        for _, row in rules_df.iterrows():
            antecedent = tuple(sorted(row["antecedent"]))
            consequent = tuple(sorted(row["consequent"]))
            rule_key = (antecedent, consequent)
            if rule_key not in seen_rules:
                cursor.execute(f"""
                    INSERT INTO {get_table_name('bundling_all')} (support, lift, confidence)
                    VALUES (%s, %s, %s)
                """, (row["support"], row["lift"], row["confidence"]))
                id_bundling = cursor.lastrowid

                for ant_item in row["antecedent"]:
                    cursor.execute(f"SELECT id_barang FROM {get_table_name('barang')} WHERE deskripsi_brg = %s", (ant_item,))
                    result = cursor.fetchone()
                    if result:
                        cursor.execute(f"""
                            INSERT IGNORE INTO {get_table_name('bundling_all', is_item_table=True)} (id_bundling, id_barang, role)
                            VALUES (%s, %s, %s)
                        """, (id_bundling, result[0], 'antecedent'))

                for con_item in row["consequent"]:
                    cursor.execute(f"SELECT id_barang FROM {get_table_name('barang')} WHERE deskripsi_brg = %s", (con_item,))
                    result = cursor.fetchone()
                    if result:
                        cursor.execute(f"""
                            INSERT IGNORE INTO {get_table_name('bundling_all', is_item_table=True)} (id_bundling, id_barang, role)
                            VALUES (%s, %s, %s)
                        """, (id_bundling, result[0], 'consequent'))

                seen_rules.add(rule_key)

        conn.commit()
        st.success(f"✅ Hasil bundling ({st.session_state.bundling_type}) berhasil disimpan ke tabel `{get_table_name('bundling_all')}` dan `{get_table_name('bundling_all', is_item_table=True)}`.")
    except Exception as e:
        st.error(f"❌ Error in run_fp_growth_and_store_to_db: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()
        spark.stop()

def run_similarity_and_store_to_db():
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT st.id_barang, b.deskripsi_brg
            FROM {get_table_name('sold_together')} st 
            JOIN {get_table_name('barang')} b ON st.id_barang = b.id_barang
        """)
        sold_data = cursor.fetchall()
        sold_df = pd.DataFrame(sold_data, columns=["id_barang", "deskripsi_brg"])

        cursor.execute(f"""
            SELECT ds.id_barang, b.deskripsi_brg
            FROM {get_table_name('deadstock')} ds
            JOIN {get_table_name('barang')} b ON ds.id_barang = b.id_barang
        """)
        dead_data = cursor.fetchall()
        dead_df = pd.DataFrame(dead_data, columns=["id_barang", "deskripsi_brg"])

        sold_deskripsi = sold_df["deskripsi_brg"].astype(str).tolist()
        dead_deskripsi = dead_df["deskripsi_brg"].astype(str).tolist()
        all_deskripsi = sold_deskripsi + dead_deskripsi

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(all_deskripsi)

        similarity_matrix = cosine_similarity(vectors)

        similarity_scores = similarity_matrix[:len(sold_df), len(sold_df):]

        threshold = 0.5
        inserted = 0

        cursor.execute(f"DELETE FROM {get_table_name('similarity')}")
        cursor.execute(f"ALTER TABLE {get_table_name('similarity')} AUTO_INCREMENT = 1")

        for i, id_barang_1 in enumerate(sold_df["id_barang"]):
            for j, id_barang_2 in enumerate(dead_df["id_barang"]):
                score = similarity_scores[i][j]
                if score > threshold:
                    cursor.execute(f"""
                        INSERT INTO {get_table_name('similarity')} (id_barang_1, id_barang_2, score)
                        VALUES (%s, %s, %s)
                    """, (id_barang_1, id_barang_2, float(score)))
                    inserted += 1

                if score == 1:
                    cursor.execute(f"SELECT 1 FROM {get_table_name('deadstock')} WHERE id_barang = %s", (id_barang_1,))
                    deadstock_1_exists = cursor.fetchone()

                    cursor.execute(f"SELECT 1 FROM {get_table_name('deadstock')} WHERE id_barang = %s", (id_barang_2,))
                    deadstock_2_exists = cursor.fetchone()

                    if deadstock_1_exists:
                        cursor.execute(f"DELETE FROM {get_table_name('deadstock')} WHERE id_barang = %s", (id_barang_1,))
                        cursor.execute(f"DELETE FROM {get_table_name('barang')} WHERE id_barang = %s", (id_barang_1,))

                    if deadstock_2_exists:
                        cursor.execute(f"DELETE FROM {get_table_name('deadstock')} WHERE id_barang = %s", (id_barang_2,))
                        cursor.execute(f"DELETE FROM {get_table_name('barang')} WHERE id_barang = %s", (id_barang_2,))

                    if deadstock_1_exists or deadstock_2_exists:
                        cursor.execute(f"""
                            DELETE FROM {get_table_name('similarity')} 
                            WHERE (id_barang_1 = %s AND id_barang_2 = %s) 
                               OR (id_barang_1 = %s AND id_barang_2 = %s)
                        """, (id_barang_1, id_barang_2, id_barang_2, id_barang_1))

        conn.commit()
        st.success(f"{inserted} similarity scores > {threshold} berhasil disimpan ke tabel `{get_table_name('similarity')}`.")
    except Exception as e:
        st.error(f"❌ Error in run_similarity_and_store_to_db: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def run_bundling_similarity_and_store_to_db():
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {get_table_name('bundling_all')}")
        bundling_data = cursor.fetchall()
        bundling_df = pd.DataFrame(bundling_data, columns=["id_bundling", "support", "lift", "confidence"])

        cursor.execute(f"SELECT id_barang_1, id_barang_2 FROM {get_table_name('similarity')}")
        similarity_data = cursor.fetchall()
        similarity_df = pd.DataFrame(similarity_data, columns=["sold_together", "deadstock"])

        cursor.execute(f"SELECT id_barang, deskripsi_brg FROM {get_table_name('barang')}")
        barang_data = cursor.fetchall()
        barang_df = pd.DataFrame(barang_data, columns=["id_barang", "deskripsi_brg"])

        if barang_df.empty or 'id_barang' not in barang_df.columns or 'deskripsi_brg' not in barang_df.columns:
            raise ValueError("Table 'barang' is empty or missing required columns 'id_barang' or 'deskripsi_brg'")

        id_to_name = dict(zip(barang_df["id_barang"], barang_df["deskripsi_brg"]))

        if not id_to_name:
            st.warning("⚠️ No items found in id_to_name mapping. Product names may not display correctly.")
        
        replaced_count = 0
        perubahan_data = []

        cursor.execute(f"DELETE FROM {get_table_name('bundling_similarity')}")
        cursor.execute(f"DELETE FROM {get_table_name('bundling_similarity', is_item_table=True)}")
        cursor.execute(f"ALTER TABLE {get_table_name('bundling_similarity')} AUTO_INCREMENT = 1")
        cursor.execute(f"DELETE FROM {get_table_name('perubahan_bundling')}")
        cursor.execute(f"ALTER TABLE {get_table_name('perubahan_bundling')} AUTO_INCREMENT = 1")

        seen_rules = set()
        for _, row in bundling_df.iterrows():
            id_bundling = int(row["id_bundling"])
            support = float(row["support"]) if pd.notna(row["support"]) else None
            lift = float(row["lift"]) if pd.notna(row["lift"]) else None
            confidence = float(row["confidence"]) if pd.notna(row["confidence"]) else None

            cursor.execute(f"SELECT id_barang, role FROM {get_table_name('bundling_all', is_item_table=True)} WHERE id_bundling = %s", (id_bundling,))
            items = cursor.fetchall()
            antecedents_old = [item[0] for item in items if item[1] == 'antecedent']
            consequents_old = [item[0] for item in items if item[1] == 'consequent']

            rule_key = (tuple(sorted(antecedents_old)), tuple(sorted(consequents_old)))
            if rule_key in seen_rules:
                continue

            antecedents_new = antecedents_old.copy()
            consequents_new = consequents_old.copy()
            replaced = False

            for i, ant_id in enumerate(antecedents_old):
                replacement = similarity_df[similarity_df["sold_together"] == ant_id]
                if not replacement.empty:
                    antecedents_new[i] = int(replacement.iloc[0]["deadstock"])
                    replaced = True
                    break

            if not replaced:
                for i, con_id in enumerate(consequents_old):
                    replacement = similarity_df[similarity_df["sold_together"] == con_id]
                    if not replacement.empty:
                        consequents_new[i] = int(replacement.iloc[0]["deadstock"])
                        replaced = True
                        break

            if replaced:
                cursor.execute(f"""
                    INSERT INTO {get_table_name('bundling_similarity')} (support, lift, confidence)
                    VALUES (%s, %s, %s)
                """, (support, lift, confidence))
                new_id_bundling = cursor.lastrowid

                for ant_id in antecedents_new:
                    cursor.execute(f"""
                        INSERT IGNORE INTO {get_table_name('bundling_similarity', is_item_table=True)} (id_bundling, id_barang, role)
                        VALUES (%s, %s, %s)
                    """, (new_id_bundling, ant_id, 'antecedent'))

                for con_id in consequents_new:
                    cursor.execute(f"""
                        INSERT IGNORE INTO {get_table_name('bundling_similarity', is_item_table=True)} (id_bundling, id_barang, role)
                        VALUES (%s, %s, %s)
                    """, (new_id_bundling, con_id, 'consequent'))

                replaced_count += 1

                cursor.execute(f"""
                    INSERT INTO {get_table_name('perubahan_bundling')} (id_bundling, id_barang_1_lama, id_barang_2_lama, id_barang_1_baru, id_barang_2_baru, waktu)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    id_bundling,
                    antecedents_old[0] if antecedents_old else None,
                    consequents_old[0] if consequents_old else None,
                    antecedents_new[0] if antecedents_new else None,
                    consequents_new[0] if consequents_new else None
                ))

                perubahan_data.append({
                    "ID Bundling": id_bundling,
                    "Antecedent Lama": ", ".join([id_to_name.get(id, f"ID {id}") for id in antecedents_old]),
                    "Consequent Lama": ", ".join([id_to_name.get(id, f"ID {id}") for id in consequents_old]),
                    "Antecedent Baru": ", ".join([id_to_name.get(id, f"ID {id}") for id in antecedents_new]),
                    "Consequent Baru": ", ".join([id_to_name.get(id, f"ID {id}") for id in consequents_new])
                })
                seen_rules.add(rule_key)

        conn.commit()
        st.success(f"{replaced_count} bundling entries ({st.session_state.bundling_type}) berhasil dimasukkan ke `{get_table_name('bundling_similarity')}` dan perubahan tercatat.")
    except Exception as e:
        st.error(f"❌ Error in run_bundling_similarity_and_store_to_db: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def run_deadstock_fpgrowth_to_db():
    try:
        conn = connect_db()
        cursor = conn.cursor()

        df_penjualan = pd.read_sql(f"""
            SELECT p.kode_nota, b.deskripsi_brg, b.id_barang 
            FROM {get_table_name('penjualan')} p 
            JOIN {get_table_name('barang')} b ON p.id_barang = b.id_barang
        """, conn)

        df_rfm = pd.read_sql(f"""
            SELECT r.id_barang, c.category 
            FROM {get_table_name('rfm')} r
            JOIN {get_table_name('category')} c ON r.id_category = c.id_category
        """, conn)

        temp = df_penjualan.groupby("kode_nota")["id_barang"].count().reset_index(name="count")
        valid_nota = temp[temp["count"] > 1]["kode_nota"]
        df_penjualan_filtered = df_penjualan[df_penjualan["kode_nota"].isin(valid_nota)]

        df_joined = df_penjualan_filtered.merge(df_rfm, on="id_barang", how="left")

        def filter_deadstock(group):
            return (group["category"] == "Deadstock (Non-Moving Products)").any()

        grouped = df_joined.groupby("kode_nota")
        filtered_groups = grouped.filter(filter_deadstock)

        filtered_groups = filtered_groups.dropna(subset=["deskripsi_brg"])
        grouped_items = filtered_groups.groupby("kode_nota")["deskripsi_brg"].apply(set).reset_index()
        grouped_items["deskripsi_brg"] = grouped_items["deskripsi_brg"].apply(list)

        spark = SparkSession.builder.appName("FPGrowth_Deadstock_DB").getOrCreate()
        df_spark = spark.createDataFrame(grouped_items.rename(columns={"deskripsi_brg": "items"}))

        params = get_fp_growth_ds_params(st.session_state.bundling_type)
        fp_growth = FPGrowth(itemsCol="items", minSupport=params["minSupport"], minConfidence=params["minConfidence"])
        model = fp_growth.fit(df_spark)

        rules_df = model.associationRules.toPandas()

        df_map = pd.read_sql(f"""
            SELECT b.deskripsi_brg, c.category 
            FROM {get_table_name('rfm')} r
            JOIN {get_table_name('barang')} b ON r.id_barang = b.id_barang
            JOIN {get_table_name('category')} c ON r.id_category = c.id_category
        """, conn)
        rfm_dict = dict(zip(df_map["deskripsi_brg"], df_map["category"]))

        def contains_deadstock(items):
            return any(rfm_dict.get(item, "") == "Deadstock (Non-Moving Products)" for item in items)

        rules_df = rules_df[rules_df.apply(
            lambda x: contains_deadstock(x["antecedent"]) or contains_deadstock(x["consequent"]),
            axis=1
        )]

        cursor.execute(f"DELETE FROM {get_table_name('bundling_ds')}")
        cursor.execute(f"DELETE FROM {get_table_name('bundling_ds', is_item_table=True)}")
        cursor.execute(f"ALTER TABLE {get_table_name('bundling_ds')} AUTO_INCREMENT = 1")

        seen_rules = set()
        inserted = 0
        for _, row in rules_df.iterrows():
            antecedent = tuple(sorted(row["antecedent"]))
            consequent = tuple(sorted(row["consequent"]))
            rule_key = (antecedent, consequent)
            if rule_key not in seen_rules:
                cursor.execute(f"""
                    INSERT INTO {get_table_name('bundling_ds')} (support, confidence, lift)
                    VALUES (%s, %s, %s)
                """, (float(row["support"]), float(row["confidence"]), float(row["lift"])))
                id_bundling = cursor.lastrowid

                for ant_item in row["antecedent"]:
                    cursor.execute(f"SELECT id_barang FROM {get_table_name('barang')} WHERE deskripsi_brg = %s", (ant_item,))
                    result = cursor.fetchone()
                    if result:
                        cursor.execute(f"""
                            INSERT IGNORE INTO {get_table_name('bundling_ds', is_item_table=True)} (id_bundling, id_barang, role)
                            VALUES (%s, %s, %s)
                        """, (id_bundling, result[0], 'antecedent'))

                for con_item in row["consequent"]:
                    cursor.execute(f"SELECT id_barang FROM {get_table_name('barang')} WHERE deskripsi_brg = %s", (con_item,))
                    result = cursor.fetchone()
                    if result:
                        cursor.execute(f"""
                            INSERT IGNORE INTO {get_table_name('bundling_ds', is_item_table=True)} (id_bundling, id_barang, role)
                            VALUES (%s, %s, %s)
                        """, (id_bundling, result[0], 'consequent'))

                inserted += 1
                seen_rules.add(rule_key)

        conn.commit()
        st.success(f"✅ {inserted} bundling FP-Growth deadstock ({st.session_state.bundling_type}) berhasil disimpan ke tabel `{get_table_name('bundling_ds')}` dan `{get_table_name('bundling_ds', is_item_table=True)}`.")
    except Exception as e:
        st.error(f"❌ Error in run_deadstock_fpgrowth_to_db: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()
        spark.stop()

def check_all_tables_have_data():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        tables = [
            "bundling_all", "bundling_similarity", "bundling_ds",
            "perubahan_bundling", "similarity", "sold_together",
            "deadstock", "penjualan", "rfm", "category", "barang"
        ]
        
        has_data = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {get_table_name(table)}")
            count = cursor.fetchone()[0]
            has_data[table] = count > 0

        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {get_table_name('bundling_all')} ba
            JOIN {get_table_name('bundling_all', is_item_table=True)} bi1 ON ba.id_bundling = bi1.id_bundling AND bi1.role = 'antecedent'
            JOIN {get_table_name('bundling_all', is_item_table=True)} bi2 ON ba.id_bundling = bi2.id_bundling AND bi2.role = 'consequent'
            LEFT JOIN {get_table_name('barang')} b1 ON bi1.id_barang = b1.id_barang
            LEFT JOIN {get_table_name('barang')} b2 ON bi2.id_barang = b2.id_barang
            WHERE b1.id_barang IS NULL OR b2.id_barang IS NULL
        """)
        invalid_bundling_all = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {get_table_name('bundling_similarity')} bs
            JOIN {get_table_name('bundling_similarity', is_item_table=True)} bi1 ON bs.id_bundling = bi1.id_bundling AND bi1.role = 'antecedent'
            JOIN {get_table_name('bundling_similarity', is_item_table=True)} bi2 ON bs.id_bundling = bi2.id_bundling AND bi2.role = 'consequent'
            LEFT JOIN {get_table_name('barang')} b1 ON bi1.id_barang = b1.id_barang
            LEFT JOIN {get_table_name('barang')} b2 ON bi2.id_barang = b2.id_barang
            WHERE b1.id_barang IS NULL OR b2.id_barang IS NULL
        """)
        invalid_bundling_similarity = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {get_table_name('bundling_ds')} bd
            JOIN {get_table_name('bundling_ds', is_item_table=True)} bi1 ON bd.id_bundling = bi1.id_bundling AND bi1.role = 'antecedent'
            JOIN {get_table_name('bundling_ds', is_item_table=True)} bi2 ON bd.id_bundling = bi2.id_bundling AND bi2.role = 'consequent'
            LEFT JOIN {get_table_name('barang')} b1 ON bi1.id_barang = b1.id_barang
            LEFT JOIN {get_table_name('barang')} b2 ON bi2.id_barang = b2.id_barang
            WHERE b1.id_barang IS NULL OR b2.id_barang IS NULL
        """)
        invalid_bundling_ds = cursor.fetchone()[0]

        has_supporting_data = all(has_data[t] for t in ["barang", "rfm", "category"])
        is_consistent = (invalid_bundling_all == 0 and invalid_bundling_similarity == 0 and invalid_bundling_ds == 0)
        
        return has_supporting_data and any(has_data[t] for t in ["bundling_all", "bundling_similarity", "bundling_ds"]) and is_consistent

    except mysql.connector.Error as e:
        print(f"❌ Gagal memeriksa database: {str(e)}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def clear_all_tables():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        tables = [
            "bundling_all", "bundling_similarity", "bundling_ds",
            "perubahan_bundling", "similarity", "sold_together",
            "deadstock", "penjualan", "rfm", "category", "barang"
        ]
        
        item_tables = [
            "bundling_all_item", "bundling_similarity_item", "bundling_ds_item"
        ]
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        for table in tables + item_tables:
            full_table_name = get_table_name(table) if table not in item_tables else get_table_name(table.split('_item')[0], is_item_table=True)
            cursor.execute(f"DELETE FROM {full_table_name}")
            cursor.execute(f"ALTER TABLE {full_table_name} AUTO_INCREMENT = 1")
            print(f"[DEBUG] Data dihapus dan auto-increment direset untuk tabel {full_table_name}")
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        conn.commit()
        print("✅ Semua data di database telah dihapus dan auto-increment direset ke 1.")
    except mysql.connector.Error as e:
        print(f"❌ Gagal menghapus data atau mereset auto-increment di database: {str(e)}")
        conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# ============== STREAMLIT WIZARD UI ==============
st.set_page_config(page_title="Bundling Produk & Segmentasi", layout="wide")
st.title("📦 Analisis Bundling Produk & Segmentasi")

# Inisialisasi session state untuk wizard
if "wizard_step" not in st.session_state:
    has_data = check_all_tables_have_data()
    st.session_state.wizard_step = 4 if has_data else 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "deadstock_file_data" not in st.session_state:
    st.session_state.deadstock_file_data = None
if "tahun_tren" not in st.session_state:
    st.session_state.tahun_tren = 1
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "selected_sheet" not in st.session_state:
    st.session_state.selected_sheet = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = check_all_tables_have_data()
if "deadstock_valid" not in st.session_state:
    st.session_state.deadstock_valid = False
if "bundling_type" not in st.session_state:
    st.session_state.bundling_type = "General"

# Fungsi untuk navigasi wizard
def next_step():
    st.session_state.wizard_step += 1
    st.write(f"[DEBUG] Advancing to step {st.session_state.wizard_step}")
    st.rerun()

def prev_step():
    st.session_state.wizard_step -= 1
    st.write(f"[DEBUG] Going back to step {st.session_state.wizard_step}")
    st.rerun()

def go_to_step_4():
    st.session_state.wizard_step = 4
    st.write(f"[DEBUG] Jumping to Step 4 to view results")
    st.rerun()

# Langkah 0: Pilih Tipe Bundling
if st.session_state.wizard_step == 0:
    st.header("Langkah 0: Pilih Tipe Bundling")
    st.write("Pilih tipe bundling atau platform untuk analisis data.")
    
    bundling_type = st.selectbox(
        "Pilih tipe bundling:",
        options=["General", "Shopee", "Tokopedia", "Lazada"],
        index=0,
        key="bundling_type_step0"
    )
    st.session_state.bundling_type = bundling_type
    
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 250px !important;
            height: 50px !important;
            font-size: 18px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Lanjut", key="step0_next"):
            st.write(f"[DEBUG] Step 0: 'Lanjut' button clicked, bundling_type={st.session_state.bundling_type}")
            next_step()
    with col3:
        if check_platform_data(bundling_type):
            if st.button("🔍 Lihat Hasil Bundling Terakhir", key="step0_to_step4"):
                st.write("[DEBUG] Step 0: 'Lihat Hasil' button clicked")
                go_to_step_4()
        else:
            st.button(
                "🔍 Lihat Hasil Bundling Terakhir",
                disabled=True,
                key="step0_to_step4_disabled",
                help="Analisis belum selesai. Silakan jalankan analisis di Langkah 3 terlebih dahulu."
            )
    
    st.write("---")
    st.write("Catatan: Pilihan tipe bundling akan memengaruhi parameter analisis seperti support dan confidence.")

# Langkah 1: Upload File
elif st.session_state.wizard_step == 1:
    st.header("Langkah 1: Upload File")
    st.write("Unggah file Excel berisi data penjualan/pembelian dan file deadstock.")
    
    st.subheader("📂 Upload File Penjualan")
    uploaded_files = st.file_uploader(
        "Pilih file Excel (.xlsx) untuk data penjualan",
        type="xlsx",
        accept_multiple_files=True,
        key="sales_files_step1"
    )
    
    platform_valid = True
    if uploaded_files:
        expected_keyword = st.session_state.bundling_type.lower()
        if st.session_state.bundling_type != "General":
            for file in uploaded_files:
                if expected_keyword not in file.name.lower():
                    platform_valid = False
                    st.warning(f"⚠️ Nama file penjualan harus mengandung '{expected_keyword}' untuk tipe bundling {st.session_state.bundling_type}.")
                    break
        else:
            platform_valid = True
    
    st.subheader("🪦 Upload File Deadstock (DS=100)")
    deadstock_file = st.file_uploader(
        "Pilih file Excel (.xlsx) untuk data deadstock",
        type="xlsx",
        key="deadstock_file_step1"
    )
    
    if deadstock_file:
        try:
            df_dead = pd.read_excel(deadstock_file)
            required_deadstock_columns = ["Deskripsi Brg", "DS"]
            missing_deadstock_columns = [col for col in required_deadstock_columns if col not in df_dead.columns]
            if missing_deadstock_columns:
                st.session_state.deadstock_valid = False
            else:
                st.session_state.deadstock_valid = True
                st.session_state.deadstock_file_data = deadstock_file
                st.success("✅ File deadstock berhasil diunggah dan memiliki kolom yang diperlukan.")
        except Exception as e:
            st.error(f"❌ Gagal memproses file deadstock: {str(e)}")
            st.session_state.deadstock_valid = False
    
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 250px !important;
            height: 50px !important;
            font-size: 18px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    container = st.container()
    with container:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if uploaded_files and deadstock_file and st.session_state.deadstock_valid and platform_valid:
                st.session_state.uploaded_files = uploaded_files
                st.success("✅ Semua file berhasil diunggah!")
                if st.button("Lanjut", key="step1_next"):
                    st.write("[DEBUG] Step 1: 'Lanjut' button clicked")
                    next_step()
            else:
                if not uploaded_files:
                    st.warning("⚠️ Harap unggah file penjualan sebelum melanjut.")
                if not deadstock_file:
                    st.warning("⚠️ Harap unggah file deadstock sebelum melanjut.")
                if deadstock_file and not st.session_state.deadstock_valid:
                    st.warning("⚠️ Harap perbaiki file deadstock untuk memastikan kolom 'Deskripsi Brg' dan 'DS' tersedia.")
                if not platform_valid:
                    st.warning(f"⚠️ Nama file penjualan harus mengandung '{st.session_state.bundling_type.lower()}' untuk tipe bundling {st.session_state.bundling_type}.")
    
    result_container = st.container()
    with result_container:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.session_state.analysis_done:
                if st.button("🔍 Lihat Hasil Bundling Terakhir", key="step1_to_step4"):
                    st.write("[DEBUG] Step 1: 'Lihat Hasil' button clicked")
                    go_to_step_4()
            else:
                st.button(
                    "🔍 Lihat Hasil Bundling Terakhir",
                    disabled=True,
                    key="step1_to_step4_disabled",
                    help="Analisis belum selesai. Silakan jalankan analisis di Langkah 3 terlebih dahulu."
                )
    
    st.write("---")
    st.write(f"Catatan: Pastikan file penjualan berisi kolom 'Kode Nota', 'Tanggal', 'Deskripsi Brg', 'Jenis Barang', dan 'Jml'. Nama file harus mengandung '{st.session_state.bundling_type.lower()}' untuk {st.session_state.bundling_type}. File deadstock harus memiliki kolom 'Deskripsi Brg' dan 'DS'.")

# Langkah 2: Pilih Periode Tren dan Pratinjau Data
elif st.session_state.wizard_step == 2:
    st.header("Langkah 2: Pilih Periode Tren & Pratinjau Data")
    
    @st.cache_data
    def load_excel_data(file, sheet_name):
        return pd.read_excel(file, sheet_name=sheet_name)

    file_names = [file.name for file in st.session_state.uploaded_files]
    selected_file_name = st.selectbox(
        "📂 Pilih file untuk ditampilkan",
        file_names,
        key="select_file_step2"
    )
    selected_file = next(file for file in st.session_state.uploaded_files if file.name == selected_file_name)
    st.session_state.selected_file = selected_file

    try:
        xls = pd.ExcelFile(selected_file)
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox(
            f"📑 Pilih sheet untuk `{selected_file_name}`",
            sheet_names,
            key="select_sheet_step2"
        )
        st.session_state.selected_sheet = selected_sheet
        
        df = load_excel_data(selected_file, selected_sheet)
        
        st.subheader("📅 Pilih Periode Tren")
        tahun_tren = st.selectbox(
            "Pilih periode tren (tahun ke belakang):",
            options=[1, 2, 3],
            index=0,
            key="tahun_tren_step2"
        )
        st.session_state.tahun_tren = tahun_tren

        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors='coerce')
        latest_date = df["Tanggal"].max()
        cutoff_date = latest_date - pd.DateOffset(years=st.session_state.tahun_tren)
        df_filtered = df[df["Tanggal"] >= cutoff_date]

        st.subheader(f"📝 Pratinjau Data: {selected_sheet} (dalam {st.session_state.tahun_tren} tahun terakhir)")
        
        rows_per_page = 10
        total_rows = len(df_filtered)
        total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)

        if "current_page_step2" not in st.session_state:
            st.session_state.current_page_step2 = 1

        pagination_info = st.empty()
        table_placeholder = st.empty()

        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            if st.button("Previous", key="prev_page_step2", disabled=(st.session_state.current_page_step2 <= 1)):
                st.session_state.current_page_step2 -= 1
        with col2:
            if st.button("Next", key="next_page_step2", disabled=(st.session_state.current_page_step2 >= total_pages)):
                st.session_state.current_page_step2 += 1
        with col3:
            selected_page = st.number_input(
                "Pilih halaman",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.current_page_step2,
                step=1,
                key="select_page_step2"
            )
            if selected_page != st.session_state.current_page_step2:
                st.session_state.current_page_step2 = selected_page

        page_number = st.session_state.current_page_step2
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)

        with pagination_info:
            st.write(f"Menampilkan baris {start_idx + 1} - {end_idx} dari {total_rows} baris (Halaman {page_number} dari {total_pages})")
        with table_placeholder:
            st.dataframe(df_filtered.iloc[start_idx:end_idx])
        
        button_container = st.container()
        with button_container:
            st.markdown(
                """
                <style>
                .stButton>button {
                    width: 250px !important;
                    height: 50px !important;
                    font-size: 18px !important;
                    margin-left: auto !important;
                    margin-right: auto !important;
                    display: block !important;
                    text-align: center !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Kembali", key="step2_prev"):
                    prev_step()
            with col2:
                if st.button("Lanjut", key="step2_next"):
                    next_step()
            with col3:
                if st.session_state.analysis_done:
                    if st.button("🔍 Lihat Hasil Bundling Terakhir", key="step2_to_step4"):
                        go_to_step_4()
                else:
                    st.button(
                        "🔍 Lihat Hasil Bundling Terakhir",
                        disabled=True,
                        key="step2_to_step4_disabled",
                        help="Analisis belum selesai. Silakan jalankan analisis di Langkah 3 terlebih dahulu."
                    )

    except Exception as e:
        st.error(f"❌ Gagal memproses file `{selected_file_name}`: {e}")

# Langkah 3: Jalankan Analisis
elif st.session_state.wizard_step == 3:
    st.header("Langkah 3: Jalankan Analisis")
    st.write("Klik tombol di bawah untuk memproses data dengan RFM, FP-Growth, similarity, dan bundling deadstock.")
    
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False

    st.markdown(
        """
        <style>
        .stButton>button {
            width: 250px !important;
            height: 50px !important;
            font-size: 18px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            text-align: center !important;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            width: 75vw !important;
            height: 50px !important;
            font-size: 18px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚗 Mulai Proses Analisis", key="run_analysis_step3", type="primary"):
        st.write("[DEBUG] Step 3: 'Mulai Proses Analisis' button clicked")
        if not st.session_state.analysis_running:
            st.session_state.analysis_running = True
            try:
                clear_all_tables()
                ensure_correct_item_table_schema()  # Pastikan skema tabel benar sebelum analisis
                selected_file = st.session_state.selected_file
                selected_sheet = st.session_state.selected_sheet
                tahun_tren = st.session_state.tahun_tren
                deadstock_file = st.session_state.deadstock_file_data
                
                xls = pd.ExcelFile(selected_file)
                df = pd.read_excel(xls, sheet_name=selected_sheet)
                
                required_columns = ["Kode Nota", "Tanggal", "Deskripsi Brg", "Jenis Barang", "Jml"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}. Please check the file format.")
                    st.session_state.analysis_running = False
                    st.stop()
                
                initial_rows = len(df)
                
                df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors='coerce')
                df = df.dropna(subset=["Tanggal"])
                dropped_rows_dates = initial_rows - len(df)
                if dropped_rows_dates > 0:
                    st.warning(f"⚠️ {dropped_rows_dates} row(s) with missing or invalid dates were removed.")

                if df.empty:
                    st.error("❌ No valid data remains after removing rows with invalid dates. Please ensure the file contains valid date data.")
                    st.session_state.analysis_running = False
                    st.stop()
                
                df = df.dropna(subset=["Kode Nota", "Deskripsi Brg", "Jenis Barang", "Jml"])
                dropped_rows_other = len(df) - (initial_rows - dropped_rows_dates)
                if dropped_rows_other > 0:
                    st.warning(f"⚠️ {dropped_rows_other} row(s) with missing values in 'Kode Nota', 'Deskripsi Brg', 'Jenis Barang', or 'Jml' were removed.")

                df["Jml"] = pd.to_numeric(df["Jml"], errors='coerce')
                initial_rows_numeric = len(df)
                df = df.dropna(subset=["Jml"])
                dropped_rows_numeric = initial_rows_numeric - len(df)
                if dropped_rows_numeric > 0:
                    st.warning(f"⚠️ {dropped_rows_numeric} row(s) with non-numeric values in 'Jml' were removed.")

                df["Jml"] = df["Jml"].astype(int)

                if df.empty:
                    st.error("❌ No valid data remains after cleaning. Please check the file for missing or invalid values.")
                    st.session_state.analysis_running = False
                    st.stop()
                
                nat_count = df["Tanggal"].isna().sum()
                if nat_count > 0:
                    st.warning(f"⚠️ Found {nat_count} NaT values in 'Tanggal' column after cleaning. Dropping these rows.")
                    df = df.dropna(subset=["Tanggal"])
                df["Tanggal"] = df["Tanggal"].astype("datetime64[ns]")

                if not isinstance(tahun_tren, int) or tahun_tren <= 0:
                    st.error(f"❌ Invalid trend period (tahun_tren={tahun_tren}). It must be a positive integer.")
                    st.session_state.analysis_running = False
                    st.stop()

                latest_date = df["Tanggal"].max()
                if pd.isna(latest_date):
                    st.error("❌ Unable to determine the latest date. The 'Tanggal' column may contain only invalid or missing values.")
                    st.session_state.analysis_running = False
                    st.stop()

                cutoff_date = latest_date - pd.DateOffset(years=tahun_tren)
                if pd.isna(cutoff_date):
                    st.error("❌ Unable to compute cutoff date. Please check the date range and trend period.")
                    st.session_state.analysis_running = False
                    st.stop()

                df_filtered = df[df["Tanggal"] >= cutoff_date]

                if df_filtered.empty:
                    st.error(f"❌ No data remains after filtering for the last {tahun_tren} year(s). Try a different trend period or check the date range in your file.")
                    st.session_state.analysis_running = False
                    st.stop()

                conn = connect_db()
                cursor = conn.cursor()

                try:
                    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                    
                    cursor.execute(f"DELETE FROM {get_table_name('penjualan')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('penjualan')} AUTO_INCREMENT = 1")
                    cursor.execute(f"DELETE FROM {get_table_name('rfm')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('rfm')} AUTO_INCREMENT = 1")
                    cursor.execute(f"DELETE FROM {get_table_name('barang')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('barang')} AUTO_INCREMENT = 1")
                    cursor.execute(f"DELETE FROM {get_table_name('category')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('category')} AUTO_INCREMENT = 1")
                    cursor.execute(f"DELETE FROM {get_table_name('deadstock')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('deadstock')} AUTO_INCREMENT = 1")
                    cursor.execute(f"DELETE FROM {get_table_name('similarity')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('similarity')} AUTO_INCREMENT = 1")
                    cursor.execute(f"DELETE FROM {get_table_name('bundling_similarity')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('bundling_similarity')} AUTO_INCREMENT = 1")
                    cursor.execute(f"DELETE FROM {get_table_name('bundling_similarity', is_item_table=True)}")
                    cursor.execute(f"ALTER TABLE {get_table_name('bundling_similarity', is_item_table=True)} AUTO_INCREMENT = 1")

                    conn.commit()

                    for _, row in df_filtered.iterrows():
                        deskripsi = str(row["Deskripsi Brg"]).strip()
                        jenis = str(row["Jenis Barang"]).strip()
                        cursor.execute(f"SELECT id_barang FROM {get_table_name('barang')} WHERE deskripsi_brg = %s", (deskripsi,))
                        if not cursor.fetchone():
                            cursor.execute(f"INSERT INTO {get_table_name('barang')} (deskripsi_brg, jenis_barang) VALUES (%s, %s)", (deskripsi, jenis))

                    conn.commit()

                    cursor.execute(f"SELECT id_barang, deskripsi_brg FROM {get_table_name('barang')}")
                    barang_map = {row[1]: row[0] for row in cursor.fetchall()}

                    for _, row in df_filtered.iterrows():
                        nota = str(row["Kode Nota"])
                        tanggal = pd.to_datetime(row["Tanggal"]).date()
                        deskripsi = str(row["Deskripsi Brg"]).strip()
                        jml = int(row["Jml"])
                        id_barang = barang_map.get(deskripsi)
                        if id_barang:
                            cursor.execute(f"""
                                INSERT INTO {get_table_name('penjualan')} (kode_nota, tanggal, id_barang, jml)
                                VALUES (%s, %s, %s, %s)
                            """, (nota, tanggal, id_barang, jml))

                    conn.commit()

                    cursor.execute(f"""
                        SELECT p.kode_nota, b.deskripsi_brg 
                        FROM {get_table_name('penjualan')} p 
                        JOIN {get_table_name('barang')} b ON p.id_barang = b.id_barang
                    """)
                    rows = cursor.fetchall()

                    df_sales = pd.DataFrame(rows, columns=["Kode Nota", "Deskripsi Brg"])

                    grouped_sales = df_sales.groupby('Kode Nota')['Deskripsi Brg'].apply(set).reset_index()
                    sold_together = grouped_sales[grouped_sales['Deskripsi Brg'].apply(len) > 1]
                    sold_together_items = set()
                    for items in sold_together['Deskripsi Brg']:
                        sold_together_items.update(items)

                    cursor.execute(f"DELETE FROM {get_table_name('sold_together')}")
                    cursor.execute(f"ALTER TABLE {get_table_name('sold_together')} AUTO_INCREMENT = 1")
                    for deskripsi in sold_together_items:
                        id_barang = barang_map.get(deskripsi)
                        if id_barang:
                            cursor.execute(f"SELECT 1 FROM {get_table_name('sold_together')} WHERE id_barang = %s", (id_barang,))
                            if not cursor.fetchone():
                                cursor.execute(f"INSERT INTO {get_table_name('sold_together')} (id_barang) VALUES (%s)", (id_barang,))

                    df_dead = pd.read_excel(deadstock_file)
                    if "DS" not in df_dead.columns or "Deskripsi Brg" not in df_dead.columns:
                        st.error("❌ File deadstock harus memiliki kolom 'Deskripsi Brg' dan 'DS'.")
                        st.session_state.analysis_running = False
                        st.stop()

                    df_dead_filtered = df_dead[df_dead["DS"] == '100']

                    for _, row in df_dead_filtered.iterrows():
                        deskripsi = str(row["Deskripsi Brg"]).strip()
                        jenis = str(row.get("Jenis Barang", "")).strip()
                        cursor.execute(f"SELECT id_barang FROM {get_table_name('barang')} WHERE deskripsi_brg = %s", (deskripsi,))
                        result = cursor.fetchone()

                        if result:
                            continue
                        else:
                            cursor.execute(f"INSERT INTO {get_table_name('barang')} (deskripsi_brg, jenis_barang) VALUES (%s, %s)", (deskripsi, jenis))
                            id_barang = cursor.lastrowid

                        cursor.execute(f"INSERT INTO {get_table_name('deadstock')} (id_barang) VALUES (%s)", (id_barang,))

                    conn.commit()

                    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

                    process_rfm_from_db(1)
                    run_fp_growth_and_store_to_db()
                    run_similarity_and_store_to_db()
                    run_bundling_similarity_and_store_to_db()
                    run_deadstock_fpgrowth_to_db()

                    st.session_state.analysis_done = True
                    st.session_state.analysis_running = False
                    st.success("✅ Analisis selesai! Lanjut ke langkah berikutnya untuk melihat hasil.")
                    st.session_state.wizard_step = 4
                    st.rerun()

                except Exception as e:
                    conn.rollback()
                    st.session_state.analysis_running = False
                    st.error(f"❌ Error during database operations: {str(e)}")
                    raise
                finally:
                    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()
                
            except Exception as e:
                st.session_state.analysis_running = False
                st.error(f"❌ Error in analysis process: {str(e)}")

    button_container = st.container()
    with button_container:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Kembali", key="step3_prev"):
                st.write("[DEBUG] Step 3: 'Kembali' button clicked")
                prev_step()
        with col3:
            if st.session_state.analysis_done:
                if st.button("🔍 Lihat Hasil Bundling Terakhir", key="step3_to_step4"):
                    st.write("[DEBUG] Step 3: 'Lihat Hasil' button clicked")
                    go_to_step_4()
            else:
                st.button(
                    "🔍 Lihat Hasil Bundling Terakhir",
                    disabled=True,
                    key="step3_to_step4_disabled",
                    help="Analisis belum selesai. Silakan jalankan analisis terlebih dahulu."
                )

# Langkah 4: Lihat Hasil
elif st.session_state.wizard_step == 4:
    st.header("Langkah 4: Hasil Analisis")
    st.write(f"[DEBUG] Entered Step 4, analysis_done={st.session_state.analysis_done}")
    
    view_platform = st.selectbox(
        "Pilih platform untuk melihat hasil:",
        options=["General", "Shopee", "Tokopedia", "Lazada"],
        index=["General", "Shopee", "Tokopedia", "Lazada"].index(st.session_state.bundling_type),
        key="view_platform_step4"
    )
    
    if not check_platform_data(view_platform):
        st.warning(f"⚠️ Data untuk platform {view_platform} belum tersedia. Silakan jalankan analisis untuk platform tersebut terlebih dahulu.")
    else:
        tab1, tab2, tab3 = st.tabs(["📦 Bundling All Product", "🤝 Bundling Item Deadstock 100%", "🪘 Bundling Deadstock"])
        
        with tab1:
            st.markdown("## 📊 Hasil General Bundling Produk")
            try:
                conn = connect_db()
                df_bundling = pd.read_sql(f"""
                    SELECT 
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'antecedent' THEN b.deskripsi_brg END SEPARATOR ', ') AS Antecedent,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'consequent' THEN b.deskripsi_brg END SEPARATOR ', ') AS Consequent,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'antecedent' THEN c.category END SEPARATOR ', ') AS Kategori_Antecedent,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'consequent' THEN c.category END SEPARATOR ', ') AS Kategori_Consequent,
                        ba.support,
                        ba.lift,
                        ba.confidence
                    FROM {get_table_name('bundling_all', view_platform)} ba
                    JOIN {get_table_name('bundling_all', view_platform, True)} bi ON ba.id_bundling = bi.id_bundling
                    JOIN {get_table_name('barang', view_platform)} b ON bi.id_barang = b.id_barang
                    LEFT JOIN {get_table_name('rfm', view_platform)} r ON b.id_barang = r.id_barang
                    LEFT JOIN {get_table_name('category')} c ON r.id_category = c.id_category
                    GROUP BY ba.id_bundling, ba.support, ba.lift, ba.confidence
                    ORDER BY ba.support DESC;
                """, conn)
                st.dataframe(df_bundling)
            except Exception as e:
                st.warning(f"Tidak bisa mengambil data bundling: {e}")
            finally:
                if 'conn' in locals():
                    conn.close()

        with tab2:
            st.markdown("## 🤝 Hasil Bundling Berdasarkan Similarity")
            try:
                conn = connect_db()
                df_bundling_similarity = pd.read_sql(f"""
                    SELECT 
                        bs.id_bundling,
                        bs.support,
                        bs.lift,
                        bs.confidence,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'antecedent' THEN b.deskripsi_brg END SEPARATOR ', ') AS Antecedent,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'consequent' THEN b.deskripsi_brg END SEPARATOR ', ') AS Consequent
                    FROM {get_table_name('bundling_similarity', view_platform)} bs
                    JOIN {get_table_name('bundling_similarity', view_platform, True)} bi ON bs.id_bundling = bi.id_bundling
                    JOIN {get_table_name('barang', view_platform)} b ON bi.id_barang = b.id_barang
                    GROUP BY bs.id_bundling, bs.support, bs.lift, bs.confidence
                    ORDER BY bs.support DESC;
                """, conn)
                st.dataframe(df_bundling_similarity)
            except Exception as e:
                st.warning(f"Tidak bisa mengambil data bundling similarity: {e}")
            finally:
                if 'conn' in locals():
                    conn.close()

        with tab3:
            st.markdown("## 📊 Hasil Bundling Produk Deadstock")
            try:
                conn = connect_db()
                df_bundling = pd.read_sql(f"""
                    SELECT 
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'antecedent' THEN b.deskripsi_brg END SEPARATOR ', ') AS Antecedent,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'consequent' THEN b.deskripsi_brg END SEPARATOR ', ') AS Consequent,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'antecedent' THEN COALESCE(c.category, 'Deadstock (Non-Moving Products)') END SEPARATOR ', ') AS Kategori_Antecedent,
                        GROUP_CONCAT(DISTINCT CASE WHEN bi.role = 'consequent' THEN COALESCE(c.category, 'Deadstock (Non-Moving Products)') END SEPARATOR ', ') AS Kategori_Consequent,
                        ba.support,
                        ba.lift,
                        ba.confidence
                    FROM {get_table_name('bundling_ds', view_platform)} ba
                    JOIN {get_table_name('bundling_ds', view_platform, True)} bi ON ba.id_bundling = bi.id_bundling
                    JOIN {get_table_name('barang', view_platform)} b ON bi.id_barang = b.id_barang
                    LEFT JOIN {get_table_name('rfm', view_platform)} r ON b.id_barang = r.id_barang
                    LEFT JOIN {get_table_name('category')} c ON r.id_category = c.id_category
                    GROUP BY ba.id_bundling, ba.support, ba.lift, ba.confidence
                    ORDER BY ba.support DESC;
                """, conn)
                st.dataframe(df_bundling)
            except Exception as e:
                st.warning(f"Tidak bisa mengambil data bundling: {e}")
            finally:
                if 'conn' in locals():
                    conn.close()

    col1, col2 = st.columns(2)
    with col2:
        if st.button("Mulai Ulang", key="restart_step4"):
            st.write("[DEBUG] Step 4: 'Mulai Ulang' button clicked")
            st.session_state.wizard_step = 0
            st.rerun()

    st.markdown("Catatan: Hasil bundling menunjukkan pasangan produk dengan metrik support, lift, dan confidence. Gunakan hasil ini untuk strategi bundling di Marketplace.")