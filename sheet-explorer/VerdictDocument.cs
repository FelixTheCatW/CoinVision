using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;

namespace SheetExplorer;

/// <summary>
/// Одна строка verdict.csv. Колонки image, kind и note; прочие колонки, если они есть,
/// сохраняются как были. Изменения kind/note сразу видны в таблице.
/// </summary>
public sealed class VerdictRow : INotifyPropertyChanged
{
    private readonly string[] _fields;
    private string _kind;
    private string _note;

    internal VerdictRow(string[] fields)
    {
        _fields = fields;
        Image = fields.Length > 0 ? fields[0] : "";
        _kind = fields.Length > 1 ? fields[1] : "";
        _note = fields.Length > 2 ? fields[2] : "";
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>Путь к изображению относительно корня проекта (первая колонка, только для чтения).</summary>
    public string Image { get; }

    /// <summary>Тип монеты: engraved / slab (вторая колонка).</summary>
    public string Kind
    {
        get => _kind;
        set
        {
            value ??= "";
            if (_kind == value)
            {
                return;
            }

            _kind = value;
            OnPropertyChanged();
        }
    }

    /// <summary>Комментарий проверяющего (третья колонка).</summary>
    public string Note
    {
        get => _note;
        set
        {
            value ??= "";
            if (_note == value)
            {
                return;
            }

            _note = value;
            OnPropertyChanged();
        }
    }

    /// <summary>Строка для записи в CSV: первые три колонки берутся из свойств, остальные — как в файле.</summary>
    internal string[] ToFields()
    {
        var fields = new string[Math.Max(3, _fields.Length)];
        Array.Copy(_fields, fields, _fields.Length);
        fields[0] = Image;
        fields[1] = Kind.Trim();
        fields[2] = Note.Trim();
        return fields;
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}

/// <summary>
/// CSV-файл verdict.csv: заголовок, строки и запись обратно в тот же файл
/// с сохранением кодировки, переводов строк и порядка строк.
/// </summary>
public sealed class VerdictDocument
{
    private readonly string _path;
    private readonly string _headerLine;
    private readonly string _newLine;
    private readonly Encoding _encoding;
    private readonly bool _trailingNewLine;

    private VerdictDocument(
        string path,
        string headerLine,
        string newLine,
        Encoding encoding,
        bool trailingNewLine,
        List<VerdictRow> rows)
    {
        _path = path;
        _headerLine = headerLine;
        _newLine = newLine;
        _encoding = encoding;
        _trailingNewLine = trailingNewLine;
        Rows = rows;
    }

    public IReadOnlyList<VerdictRow> Rows { get; }

    public string Path => _path;

    public static VerdictDocument Load(string path)
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
        var headerFields = ParseLine(headerLine);
        if (headerFields.Length < 3)
        {
            throw new InvalidDataException(
                $"Ожидались колонки image,kind,note, а в заголовке: «{headerLine}».");
        }

        var rows = new List<VerdictRow>(lines.Count - 1);
        foreach (var line in lines.Skip(1))
        {
            if (line.Trim().Length == 0)
            {
                continue;
            }

            rows.Add(new VerdictRow(ParseLine(line)));
        }

        return new VerdictDocument(path, headerLine, newLine, encoding, trailingNewLine, rows);
    }

    /// <summary>Перезаписывает файл целиком: сначала во временный файл, затем подменяет оригинал.</summary>
    public void Save()
    {
        var builder = new StringBuilder();
        builder.Append(_headerLine).Append(_newLine);

        foreach (var row in Rows)
        {
            var fields = row.ToFields();
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
