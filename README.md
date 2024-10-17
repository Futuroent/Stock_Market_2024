# Apple Stock Performance & Sentiment Analysis

## Projektübersicht
 
Dieses Projekt analysiert die Performance der Apple-Aktie (AAPL) in einem bestimmten Zeitraum und stellt diese Daten im Vergleich zu Sentiment-Analysen von Nachrichtenartikeln dar. Ziel des Projekts ist es, visuelle Einblicke in die Aktienperformance von Apple zu gewinnen und zu untersuchen, wie die Nachrichtenlage die Aktienbewegungen beeinflusst. Die Ergebnisse werden in **Tableau** visualisiert, um interaktive Dashboards zu erstellen, die für Investoren und Marktanalysten hilfreich sind.

## Motivation

Die Idee hinter diesem Projekt ist es, die Zusammenhänge zwischen Nachrichten-Sentiment und Aktienbewegungen zu verstehen. Besonders bei großen Tech-Firmen wie Apple kann die Berichterstattung in den Medien einen großen Einfluss auf die Performance der Aktie haben. Durch die Verknüpfung von Nachrichtenartikeln mit ihrer Sentiment-Analyse und den entsprechenden Aktienkursbewegungen können Trends und Einflussfaktoren erkannt werden.

## Features

- **Datenquellen**:
  - Historische Apple-Aktienkurse von [Yahoo Finance](https://finance.yahoo.com).
  - Nachrichtenartikel über Apple aus verschiedenen Quellen via News API.
  - Sentiment-Analyse der Artikel mit Hilfe eines auf Finanznachrichten spezialisierten Modells.

- **Dashboards in Tableau**:
  - **Stock Price Over Time**: Ein Dashboard, das die Entwicklung des Aktienkurses über verschiedene Zeiträume zeigt (Monate, Quartale, Jahre).
  - **Sentiment Analysis Impact**: Eine Übersicht der Nachrichten und ihrer Sentiment-Wertung im Vergleich zu den entsprechenden Aktienkursen.
  - **Volume vs. Price**: Eine Korrelation zwischen Handelsvolumen und Aktienkursen, um mögliche Volatilitäten zu analysieren.
  - **Real-Time Updates**: Das System kann stündlich real-time Daten über die Apple-Aktie und aktuelle Nachrichten sammeln und in einem separaten Dashboard visualisieren.

## Verwendete Technologien

- **Programmiersprache**: Python
- **Datenquellen**: 
  - Yahoo Finance API für historische Aktienkurse
  - News API für Nachrichtenartikel
  - Hugging Face Transformers (Modell zur Sentiment-Analyse)
- **Datenbank**: SQLite zur Speicherung der Sentiment-Daten und Aktienkurse.
- **Visualisierungstool**: Tableau

## Setup & Installation

### 1. Voraussetzungen

Um dieses Projekt auszuführen, benötigst du die folgenden Abhängigkeiten:
- **Python 3.x**
- **Tableau Public** (kostenloser Tableau-Desktop für Visualisierung)
- **Pandas**: Für die Datenbearbeitung
- **SQLite**: Zum Speichern der Daten
- **Yahoo Finance API**: Für den Abruf der Aktienkurse
- **News API**: Für den Abruf der Nachrichtenartikel
- **Hugging Face Transformers**: Für die Sentiment-Analyse

Du kannst die erforderlichen Python-Bibliotheken mit folgendem Befehl installieren:

```bash
pip install pandas yfinance sqlite3 requests transformers
```

### 2. Projektinstallation

1. Klone dieses Repository auf deinen lokalen Computer:

   ```bash
   git clone <repository_url>
   ```

2. Wechsel in das Projektverzeichnis:

   ```bash
   cd apple-stock-sentiment-analysis
   ```

3. Installiere die notwendigen Abhängigkeiten:

   ```bash
   pip install -r requirements.txt
   ```

4. Richte die SQLite-Datenbank ein:

   ```bash
   python setup_database.py
   ```

5. Starte die Python-Skripte, um Daten zu sammeln und zu analysieren:

   ```bash
   python collect_stock_data.py
   python sentiment_analysis.py
   ```

6. Öffne die Tableau-Dateien (`*.twbx`), um die Dashboards zu visualisieren und interaktiv zu erkunden.

## Datenquellen

- **AAPL Stock Data**: Die Aktienkurse werden mithilfe der Yahoo Finance API abgerufen. Hier werden Kursdaten, wie "Open", "Close", "High", "Low" und "Volume", gespeichert.
- **Sentiment Data**: Nachrichten über Apple werden über die News API gesammelt und mithilfe eines Sentiment-Modells analysiert, das die Stimmung als positiv, negativ oder neutral klassifiziert.

## Tableau Dashboards

### 1. **Stock Price Over Time**
   - **Daten**: AAPL Stock Data (Adj Close)
   - **X-Achse**: Zeitachse (Jahre, Quartale, Monate)
   - **Y-Achse**: Angepasster Schlusskurs (Adj Close)
   - **Beschreibung**: Dieses Dashboard visualisiert die Entwicklung des Apple-Aktienkurses über einen festgelegten Zeitraum. Es zeigt Trends und wichtige Punkte im Zeitverlauf.

### 2. **Sentiment Analysis Impact**
   - **Daten**: Sentiment-Daten, AAPL Stock Data
   - **X-Achse**: Datum
   - **Y-Achse**: Aktienkurs (Adj Close)
   - **Beschreibung**: Dieses Dashboard vergleicht die Stimmungen von Nachrichtenartikeln mit den Kursbewegungen. Es zeigt, wie positive, negative oder neutrale Nachrichten den Aktienkurs beeinflusst haben.

### 3. **Volume vs. Price**
   - **Daten**: AAPL Stock Data
   - **X-Achse**: Handelsvolumen (Volume)
   - **Y-Achse**: Aktienkurs (Adj Close)
   - **Beschreibung**: Hier wird die Korrelation zwischen Handelsvolumen und Aktienkurs visualisiert. Dieses Dashboard ist nützlich, um Spitzen im Volumen zu identifizieren und ihre Auswirkungen auf den Preis zu analysieren.

### 4. **Real-Time Updates**
   - **Daten**: Echtzeit-Daten von Yahoo Finance und Sentiment-Analysen.
   - **Beschreibung**: Dieses Dashboard aktualisiert stündlich und zeigt die aktuellen Kursbewegungen und Nachrichten-Sentiments in Echtzeit an. Es ist besonders nützlich, um kurzfristige Trends zu erkennen.

## Weitere Entwicklungen

In zukünftigen Iterationen könnte das Projekt um weitere Funktionen erweitert werden:
- **Erweiterung der API**: Hinzufügen von zusätzlichen Nachrichtenquellen und Aktien, um breitere Marktanalysen zu ermöglichen.
- **Vorhersagemodelle**: Implementierung von Machine Learning-Modellen zur Vorhersage von Aktienbewegungen basierend auf Nachrichtenstimmung.
- **Optimierte Dashboards**: Erweiterung der Interaktivität und Datenvisualisierungen.

## Fazit

Dieses Projekt bietet eine umfassende Analyse der Apple-Aktienkurse im Vergleich zu Nachrichten-Sentiments und zeigt auf, wie Nachrichten die Kursentwicklung beeinflussen können. Es stellt sowohl langfristige Trends als auch Echtzeitdaten bereit und ermöglicht es den Benutzern, tiefere Einblicke in die Faktoren zu gewinnen, die Aktienkurse beeinflussen.
