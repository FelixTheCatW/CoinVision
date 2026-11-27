using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;

namespace BoxesExplorer;

/// <summary>
/// Одна строка date_boxes.csv: image, x1..y4, kind, year. Точки области читаются и пишутся
/// в восемь колонок, kind — в одноимённую колонку, остальные колонки не трогаются.
/// </summary>
public sealed class DateBoxesRow : INotifyPropertyChanged
{
    private readonly string[] _fields;
    private readonly int _imageIndex;
    private readonly int _kindIndex;
    private readonly int _yearIndex;
    private readonly int[] _quadIndices;

    internal DateBoxesRow(
        string[] fields,
        int imageIndex,
        int kindIndex,
        int yearIndex,
        int[] quadIndices)
    {
        _fields = fields;
        _imageIndex = imageIndex;
        _kindIndex = kindIndex;
        _yearIndex = yearIndex;
        _quadIndices = quadIndices;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>Путь к изображению относительно корня проекта (только для чтения).</summary>
    public string Image => _fields[_imageIndex];

    /// <summary>Год (только для чтения: он приходит из манифеста).</summary>
    public string Year => _fields[_yearIndex];

    /// <summary>Тип монеты: engraved / slab.</summary>
    public string Kind
    {
        get => _fields[_kindIndex];
        set
        {
            value ??= "";
            if (_fields[_kindIndex] == value)
            {
                return;
            }

            _fields[_kindIndex] = value;
            OnPropertyChanged();
        }
    }

    /// <summary>Есть ли в строке полный набор из восьми координат.</summary>
    public bool HasBox => TryGetQuad(out _);

    /// <summary>Координаты одной строкой для таблицы: x1,y1,x2,y2,x3,y3,x4,y4.</summary>
    public string Box
    {
        get
        {
            if (!TryGetQuad(out var quad))
            {
                return "";
            }

            return string.Join(",", quad.Select(point => string.Create(
                CultureInfo.InvariantCulture, $"{point.X},{point.Y}")));
        }
    }

    public bool TryGetQuad(out (int X, int Y)[] points)
    {
        points = new (int X, int Y)[4];

        for (var i = 0; i < 4; i++)
        {
            if (!int.TryParse(_fields[_quadIndices[i * 2]], NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out var x) ||
                !int.TryParse(_fields[_quadIndices[(i * 2) + 1]], NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out var y))
            {
                points = [];
                return false;
            }

            points[i] = (x, y);
        }

        return true;
    }

    /// <summary>Записывает четыре точки в колонки x1..y4.</summary>
    internal void SetQuad(IReadOnlyList<(int X, int Y)> points)
    {
        for (var i = 0; i < 4 && i < points.Count; i++)
        {
            _fields[_quadIndices[i * 2]] = points[i].X.ToString(CultureInfo.InvariantCulture);
            _fields[_quadIndices[(i * 2) + 1]] = points[i].Y.ToString(CultureInfo.InvariantCulture);
        }

        OnPropertyChanged(nameof(Box));
        OnPropertyChanged(nameof(HasBox));
    }

    /// <summary>Убирает случайные пробелы в kind перед записью в файл.</summary>
    internal void TrimKind() => _fields[_kindIndex] = Kind.Trim();

    internal string[] Fields => _fields;

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}

/// <summary>
/// CSV-файл с разметкой четырёхугольников: заголовок, строки и запись обратно в тот же файл
/// с сохранением кодировки, переводов строк, порядка строк и необязательных колонок.
/// </summary>
public sealed class DateBoxesDocument
{
    private static readonly string[] QuadColumns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"];

    private readonly string _path;
    private readonly string _headerLine;
    private readonly string _newLine;
    private readonly Encoding _encoding;
    private readonly bool _trailingNewLine;

    private DateBoxesDocument(
        string path,
        string headerLine,
        string newLine,
        Encoding encoding,
        bool trailingNewLine,
        List<DateBoxesRow> rows)
    {
        _path = path;
        _headerLine = headerLine;
        _newLine = newLine;
        _encoding = encoding;
        _trailingNewLine = trailingNewLine;
        Rows = rows;
    }

    public IReadOnlyList<DateBoxesRow> Rows { get; }

    public string Path => _path;

    public static DateBoxesDocument Load(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var (encoding, bomLength) = DetectEncoding(bytes);
        var text = encoding.GetString(bytes, bomLength, bytes.Length - bomLength);

        var newLine = text.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
        var trailingNewLine = text.EndsWith('\n');
        var lines = SplitLines(text);

        if (lines.Count == 0)
        {
            throw new InvalidDataException("Файл пуст.");
        }

        var headerLine = lines[0];
        var header = ParseLine(headerLine);

        var imageIndex = IndexOf(header, "image");
        var kindIndex = IndexOf(header, "kind");
        var yearIndex = IndexOf(header, "year");
        if (imageIndex < 0 || kindIndex < 0 || yearIndex < 0)
        {
            throw new InvalidDataException(
                "Ожидались колонки image, x1..y4, kind, year, а в заголовке: «" + headerLine + "».");
        }

        var quadIndices = new int[QuadColumns.Length];
        for (var i = 0; i < QuadColumns.Length; i++)
        {
            quadIndices[i] = IndexOf(header, QuadColumns[i]);
            if (quadIndices[i] < 0)
            {
                throw new InvalidDataException(
                    "В заголовке нет колонки «" + QuadColumns[i] + "»: «" + headerLine + "».");
            }
        }

        var rows = new List<DateBoxesRow>(lines.Count - 1);
        foreach (var line in lines.Skip(1))
        {
            if (line.Trim().Length == 0)
            {
                continue;
            }

            var fields = ParseLine(line);
            if (fields.Length < header.Length)
            {
                Array.Resize(ref fields, header.Length);
                for (var i = 0; i < fields.Length; i++)
                {
                    fields[i] ??= "";
                }
            }

            rows.Add(new DateBoxesRow(fields, imageIndex, kindIndex, yearIndex, quadIndices));
        }

        return new DateBoxesDocument(path, headerLine, newLine, encoding, trailingNewLine, rows);
    }

    /// <summary>Перезаписывает файл целиком: сначала во временный файл, затем подменяет оригинал.</summary>
    public void Save()
    {
        var builder = new StringBuilder();
        builder.Append(_headerLine).Append(_newLine);

        foreach (var row in Rows)
        {
            row.TrimKind();
            var fields = row.Fields;
            for (var i = 0; i < fields.Length; i++)
            {
                if (i > 0)
                {
                    builder.Append(',');
                }

                builder.Append(FormatField(fields[i]));
            }

            builder.Append(_newLine);
        }

        if (!_trailingNewLine &&
            builder.Length >= _newLine.Length &&
            builder.ToString(builder.Length - _newLine.Length, _newLine.Length) == _newLine)
        {
            builder.Length -= _newLine.Length;
        }

        var temporary = _path + ".tmp";
        File.WriteAllText(temporary, builder.ToString(), _encoding);
        File.Move(temporary, _path, overwrite: true);
    }

    private static int IndexOf(string[] header, string name)
    {
        for (var i = 0; i < header.Length; i++)
        {
            if (string.Equals(header[i].Trim(), name, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>Поле пишется в кавычках, если содержит запятую, кавычку или перевод строки.</summary>
    private static string FormatField(string value)
    {
        var needsQuotes = value.Contains(',') || value.Contains('"') ||
                          value.Contains('\n') || value.Contains('\r');
        return needsQuotes ? "\"" + value.Replace("\"", "\"\"") + "\"" : value;
    }

    /// <summary>Разбор строки CSV с поддержкой кавычек и удвоенных кавычек внутри них.</summary>
    private static string[] ParseLine(string line)
    {
        var fields = new List<string>();
        var field = new StringBuilder();
        var inQuotes = false;

        for (var i = 0; i < line.Length; i++)
        {
            var ch = line[i];

            if (inQuotes)
            {
                if (ch == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        field.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    field.Append(ch);
                }

                continue;
            }

            switch (ch)
            {
                case '"':
                    inQuotes = true;
                    break;
                case ',':
                    fields.Add(field.ToString());
                    field.Clear();
                    break;
                default:
                    field.Append(ch);
                    break;
            }
        }

        fields.Add(field.ToString());
        return fields.ToArray();
    }

    private static List<string> SplitLines(string text)
    {
        var lines = new List<string>();
        var start = 0;

        for (var i = 0; i < text.Length; i++)
        {
            if (text[i] != '\n')
            {
                continue;
            }

            var end = i;
            if (end > start && text[end - 1] == '\r')
            {
                end--;
            }

            lines.Add(text[start..end]);
            start = i + 1;
        }

        if (start < text.Length)
        {
            lines.Add(text[start..]);
        }

        return lines;
    }

    private static (Encoding Encoding, int BomLength) DetectEncoding(byte[] bytes)
    {
        if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
        {
            return (new UTF8Encoding(true), 3);
        }

        if (bytes.Length >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE)
        {
            return (new UnicodeEncoding(false, true), 2);
        }

        if (bytes.Length >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF)
        {
            return (new UnicodeEncoding(true, true), 2);
        }

        return (new UTF8Encoding(false), 0);
    }
}
