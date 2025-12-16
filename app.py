import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Повторяет предобработку времени и категориальных признаков из main.ipynb
    для входного CSV перед предсказанием.
    Ожидается тот же набор колонок, что и при обучении.
    """
    df = df.copy()

    # Название колонки окончания полива с пробелом как в ноутбуке
    end_col = "Окончание  полива "

    # Преобразуем год и неделю в дату начала недели
    week_start = pd.to_datetime(
        df["год"].astype("Int64").astype(str).str.zfill(4)
        + df["неделя"].astype("Int64").astype(str).str.zfill(2)
        + "1",
        format="%G%V%u",
        errors="coerce",
    )

    # Смещения по времени начала и окончания полива
    start_delta = pd.to_timedelta(df["Начало полива"].astype(str), errors="coerce")
    end_delta = pd.to_timedelta(df[end_col].astype(str), errors="coerce")

    # Приводим к datetime (год+неделя+время)
    df["Начало полива"] = week_start + start_delta
    df[end_col] = week_start + end_delta

    # Строковые категориальные признаки без NaN
    start_str = df["Начало полива"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("missing")
    end_str = df[end_col].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("missing")

    week_str = df["неделя"].fillna(-1).astype(int).astype(str)
    year_str = df["год"].fillna(-1).astype(int).astype(str)

    df["Начало полива_cat"] = start_str
    df[f"{end_col}_cat"] = end_str  # важно сохранить пробел в названии
    df["неделя_cat"] = week_str
    df["год_cat"] = year_str

    # Удаляем колонку, которая дропалась при обучении (если она есть)
    if "dt_начало_полива" in df.columns:
        df = df.drop("dt_начало_полива", axis=1)

    # Удаляем колонки, которые не использовались как признаки
    drop_cols = ["Урожайность (реальный)", "Начало полива", end_col]
    df_features = df.drop(columns=drop_cols, errors="ignore")

    return df_features


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("tomato.pth")
    return model


def main():
    st.title("Tomato Yield Predictor")
    st.write("Загрузите CSV-файл с признаками, и приложение вернёт предсказания модели.")

    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл", type=["csv"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Не удалось прочитать CSV: {e}")
            return

        if data.empty:
            st.warning("Загруженный файл пустой.")
            return

        st.subheader("Входные данные")
        st.dataframe(data.head())

        # Предобработка как в main.ipynb
        try:
            data_proc = preprocess_input(data)
        except KeyError as e:
            st.error(
                f"В CSV не хватает нужной колонки для предобработки: {e}. "
                f"Убедитесь, что структура файла такая же, как при обучении."
            )
            return
        except Exception as e:
            st.error(f"Ошибка при предобработке данных: {e}")
            return

        model = load_model()
        try:
            preds = model.predict(data_proc)
        except Exception as e:
            st.error(
                f"Ошибка при вычислении предсказаний. Проверьте формат признаков. Детали: {e}"
            )
            return

        # Только один столбец с предсказанием
        result_df = pd.DataFrame({"prediction": preds})

        st.subheader("Результаты предсказания (только prediction)")
        st.dataframe(result_df.head(100))

        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Скачать результат в CSV",
            data=csv_out,
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Пожалуйста, загрузите CSV-файл.")


if __name__ == "__main__":
    main()

