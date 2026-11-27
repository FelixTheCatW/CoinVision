using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace BoxesExplorer;

/// <summary>
/// Разметка date_boxes.csv: слева изображение выбранной строки в масштабе 1:1 с маской области,
/// справа таблица. Q ставит четыре точки (x1..y4), 1/2 пишут kind, стрелки меняют строку,
/// кнопка «Сохранить» записывает файл.
/// </summary>
public partial class MainWindow : Window
{
    private const string CsvRelativePath = @"data\annotations\date_boxes.csv";
    private const int MaxPoints = 4;
    private const int ImageCacheLimit = 4;
    private const double MarkerArm = 6;

    private static readonly string[] CanonicalKinds = ["engraved", "slab"];

    private static readonly Brush[] PointBrushes =
    {
        CreateBrush(Color.FromRgb(0x35, 0xD0, 0x7F)),
        CreateBrush(Color.FromRgb(0xFF, 0xA0, 0x00)),
        CreateBrush(Color.FromRgb(0x4F, 0xA3, 0xFF)),
        CreateBrush(Color.FromRgb(0xD4, 0x6B, 0xFF)),
    };

    private static readonly Brush SelectionStrokeBrush = CreateBrush(Color.FromRgb(0x00, 0xE5, 0xFF));
    private static readonly Brush SelectionFillBrush = CreateBrush(Color.FromArgb(0x33, 0x00, 0xE5, 0xFF));
    private static readonly Brush PreviewStrokeBrush = CreateBrush(Color.FromArgb(0xB0, 0x00, 0xE5, 0xFF));

    /// <summary>Точки области в порядке нажатия Q (1-4).</summary>
    private readonly List<(int X, int Y)> _points = new();
    private readonly ObservableCollection<string> _kindOptions = new();
    private readonly Dictionary<string, BitmapSource> _imageCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _cacheOrder = new();

    private DateBoxesDocument? _document;
    private string _root = "";
    private BitmapSource? _image;
    private (int X, int Y)? _hoverPoint;

    private string _rowText = "";
    private string _cursorText = "—";
    private string? _message;
    private bool _dirty;
    private bool _editing;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
    }

    /// <summary>Подсказки для выпадающего списка колонки kind.</summary>
    public ObservableCollection<string> KindOptions => _kindOptions;

    // ---------- загрузка файла ----------

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            LoadDocument();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "boxes-explorer", MessageBoxButton.OK, MessageBoxImage.Error);
            Close();
        }
    }

    private void LoadDocument()
    {
        _root = FindProjectRoot() ?? throw new InvalidOperationException(
            "Не удалось найти файл " + CsvRelativePath + ".\n\n" +
            "Программа ищет его вверх по каталогам от своего exe, поэтому запускать её нужно внутри " +
            "папки проекта CoinVision.");

        var csvPath = Path.Combine(_root, CsvRelativePath);
        var document = DateBoxesDocument.Load(csvPath);

        foreach (var row in document.Rows)
        {
            row.PropertyChanged += OnRowChanged;
        }

        _document = document;
        Table.ItemsSource = document.Rows;
        FileText.Text = csvPath;
        SaveStateText.Text = "Файл загружен";
        RebuildOptions();
        UpdateCounters();

        if (document.Rows.Count > 0)
        {
            var start = 0;
            for (var i = 0; i < document.Rows.Count; i++)
            {
                if (!document.Rows[i].HasBox)
                {
                    start = i;
                    break;
                }
            }

            Table.SelectedIndex = start;
            Table.ScrollIntoView(Table.SelectedItem);
            Table.Focus();
        }
        else
        {
            ClearImage("Файл пуст.");
        }
    }

    /// <summary>Ищет корень проекта: каталог, в котором лежит data\annotations\date_boxes.csv.</summary>
    private static string? FindProjectRoot()
    {
        foreach (var start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            DirectoryInfo? directory;
            try
            {
                directory = new DirectoryInfo(start);
            }
            catch (ArgumentException)
            {
                continue;
            }

            while (directory is not null)
            {
                if (File.Exists(Path.Combine(directory.FullName, CsvRelativePath)))
                {
                    return directory.FullName;
                }

                directory = directory.Parent;
            }
        }

        return null;
    }

    // ---------- изменения и сохранение ----------

    private void OnRowChanged(object? sender, PropertyChangedEventArgs e)
    {
        MarkDirty();
        UpdateCounters();
    }

    private void MarkDirty()
    {
        _dirty = true;
        SaveStateText.Text = "Не сохранено…";
    }

    private void OnSaveClick(object sender, RoutedEventArgs e) => Save();

    private bool Save()
    {
        if (_document is null)
        {
            return true;
        }

        CommitPendingEdit();

        try
        {
            _document.Save();
            _dirty = false;
            SaveStateText.Text = $"Сохранено в {DateTime.Now:HH:mm:ss}";
            _message = "Файл сохранён: " + _document.Path;
            RebuildOptions();
            UpdateStatus();
            return true;
        }
        catch (Exception ex)
        {
            SaveStateText.Text = "Ошибка сохранения";
            MessageBox.Show(
                this,
                ex.Message + "\n\nФайл: " + _document.Path,
                "Не удалось сохранить date_boxes.csv",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return false;
        }
    }

    private void CommitPendingEdit()
    {
        if (!_editing)
        {
            return;
        }

        Table.CommitEdit(DataGridEditingUnit.Cell, true);
        Table.CommitEdit(DataGridEditingUnit.Row, true);
    }

    private void OnWindowClosing(object? sender, CancelEventArgs e)
    {
        if (_document is null || !_dirty)
        {
            return;
        }

        var answer = MessageBox.Show(
            this,
            "В файле есть несохранённые изменения. Сохранить перед закрытием?",
            "boxes-explorer",
            MessageBoxButton.YesNoCancel,
            MessageBoxImage.Question);

        if (answer == MessageBoxResult.Cancel)
        {
            e.Cancel = true;
        }
        else if (answer == MessageBoxResult.Yes && !Save())
        {
            e.Cancel = true;
        }
    }

    // ---------- клавиатура ----------

    private void OnPreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.S && Keyboard.Modifiers == ModifierKeys.Control)
        {
            e.Handled = true;
            Save();
            return;
        }

        if (Keyboard.Modifiers != ModifierKeys.None || _editing || IsTextInputFocused())
        {
            return;
        }

        switch (e.Key)
        {
            case Key.Q:
                e.Handled = true;
                RememberPoint();
                return;
            case Key.D1:
            case Key.NumPad1:
                e.Handled = true;
                ApplyKindAndAdvance(0);
                return;
            case Key.D2:
            case Key.NumPad2:
                e.Handled = true;
                ApplyKindAndAdvance(1);
                return;
            case Key.Up:
                e.Handled = true;
                MoveSelection(-1);
                return;
            case Key.Down:
                e.Handled = true;
                MoveSelection(1);
                return;
            case Key.Escape:
                e.Handled = true;
                ClearDraft();
                return;
        }
    }

    private static bool IsTextInputFocused() =>
        Keyboard.FocusedElement is TextBoxBase or ComboBox;

    /// <summary>1/2: подставить вариант из списка kind и перейти на строку ниже.</summary>
    private void ApplyKindAndAdvance(int index)
    {
        if (index >= _kindOptions.Count)
        {
            return;
        }

        if (Table.SelectedItem is not DateBoxesRow row)
        {
            _message = "Сначала выделите строку.";
            UpdateStatus();
            return;
        }

        row.Kind = _kindOptions[index];
        _message = $"kind = {row.Kind}";
        MoveSelection(1);
        UpdateStatus();
    }

    private void MoveSelection(int delta)
    {
        if (Table.Items.Count == 0)
        {
            return;
        }

        var current = Table.SelectedIndex < 0 ? 0 : Table.SelectedIndex;
        var next = Math.Clamp(current + delta, 0, Table.Items.Count - 1);
        if (next == Table.SelectedIndex)
        {
            return;
        }

        Table.SelectedIndex = next;
        Table.ScrollIntoView(Table.SelectedItem);
    }

    private void OnBeginningEdit(object? sender, DataGridBeginningEditEventArgs e) => _editing = true;

    private void OnCellEditEnding(object? sender, DataGridCellEditEndingEventArgs e)
    {
        _editing = false;
        Dispatcher.BeginInvoke(
            System.Windows.Threading.DispatcherPriority.Background,
            new Action(RebuildOptions));
    }

    // ---------- точки области ----------

    private void RememberPoint()
    {
        if (_image is null)
        {
            _message = "Сначала выделите строку с изображением.";
            UpdateStatus();
            return;
        }

        if (Table.SelectedItem is not DateBoxesRow row)
        {
            _message = "Сначала выделите строку.";
            UpdateStatus();
            return;
        }

        // Четыре точки уже заданы — Q начинает новую область.
        if (_points.Count >= MaxPoints)
        {
            _points.Clear();
        }

        if (!TryGetPixelUnderCursor(out var point))
        {
            _message = "Курсор вне изображения — наведите указатель на картинку и нажмите Q.";
            UpdateStatus();
            return;
        }

        _points.Add(point);

        if (_points.Count == MaxPoints)
        {
            row.SetQuad(_points);
            var text = FormatPoints(_points);
            _message = CopyToClipboard(text)
                ? $"Точки записаны и скопированы: {text}"
                : $"Точки записаны: {text}";
        }
        else
        {
            _message = $"Точка {_points.Count} = {point.X}, {point.Y}. Дальше Q.";
        }

        RedrawOverlay();
        UpdateStatus();
    }

    private void ClearDraft()
    {
        _points.Clear();
        _message = _image is null
            ? null
            : "Черновик области снят. Сохранённые координаты строки не изменились.";
        RedrawOverlay();
        UpdateStatus();
    }

    private bool TryGetPixelUnderCursor(out (int X, int Y) point)
    {
        point = default;
        if (_image is null)
        {
            return false;
        }

        var position = Mouse.GetPosition(ImageView);

        // Позиция ровно на правой/нижней границе относится к последнему пикселю.
        if (position.X < -0.5 || position.Y < -0.5 ||
            position.X > _image.PixelWidth || position.Y > _image.PixelHeight)
        {
            return false;
        }

        var x = Math.Clamp((int)Math.Floor(position.X), 0, _image.PixelWidth - 1);
        var y = Math.Clamp((int)Math.Floor(position.Y), 0, _image.PixelHeight - 1);
        point = (x, y);
        return true;
    }

    private static bool CopyToClipboard(string text)
    {
        try
        {
            Clipboard.SetText(text);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private static string FormatPoints(IEnumerable<(int X, int Y)> points) =>
        string.Join(",", points.Select(point => string.Create(
            CultureInfo.InvariantCulture, $"{point.X},{point.Y}")));

    // ---------- мышь ----------

    private void OnImageHostMouseMove(object sender, MouseEventArgs e)
    {
        if (_image is null)
        {
            return;
        }

        var previewVisible = _points.Count is > 0 and < MaxPoints;

        if (TryGetPixelUnderCursor(out var point))
        {
            _hoverPoint = point;
            _cursorText = string.Create(CultureInfo.InvariantCulture, $"{point.X}, {point.Y}");
        }
        else
        {
            _hoverPoint = null;
            _cursorText = "—";
        }

        if (previewVisible)
        {
            RedrawOverlay();
        }

        UpdateStatus();
    }

    private void OnImageHostMouseLeave(object sender, MouseEventArgs e)
    {
        _hoverPoint = null;
        _cursorText = "—";

        if (_points.Count is > 0 and < MaxPoints)
        {
            RedrawOverlay();
        }

        UpdateStatus();
    }

    // ---------- выбранная строка и изображение ----------

    private void OnSelectionChanged(object sender, SelectionChangedEventArgs e) => ShowSelectedRow();

    private void ShowSelectedRow()
    {
        if (_document is null)
        {
            return;
        }

        _points.Clear();
        _hoverPoint = null;
        _message = null;

        if (Table.SelectedItem is not DateBoxesRow row)
        {
            ClearImage("Выберите строку в таблице —\nизображение откроется здесь в масштабе 1:1");
            _rowText = "";
            RedrawOverlay();
            UpdateStatus();
            return;
        }

        var relative = row.Image.Trim();
        var prefix = $"Строка {Table.SelectedIndex + 1} из {_document.Rows.Count} · ";

        if (relative.Length == 0)
        {
            ClearImage("В колонке image этой строки нет пути к изображению.");
            _rowText = prefix + "путь не задан";
            RedrawOverlay();
            UpdateStatus();
            return;
        }

        var full = ResolveImagePath(relative);
        if (!File.Exists(full))
        {
            ClearImage("Файл не найден:\n" + relative);
            _rowText = prefix + relative + " · файл не найден";
            RedrawOverlay();
            UpdateStatus();
            return;
        }

        BitmapSource image;
        try
        {
            image = LoadImage(full);
        }
        catch (Exception ex)
        {
            ClearImage("Не удалось открыть изображение:\n" + ex.Message);
            _rowText = prefix + relative;
            RedrawOverlay();
            UpdateStatus();
            return;
        }

        _image = image;
        ImageView.Source = image;
        Placeholder.Visibility = Visibility.Collapsed;
        Scroll.ScrollToHome();

        if (row.TryGetQuad(out var quad))
        {
            _points.AddRange(quad);
            _message = "В строке уже есть область. Q начнёт новую.";
        }

        Title = $"{Path.GetFileName(full)} — boxes-explorer";
        _rowText = $"{prefix}{relative} · {image.PixelWidth}×{image.PixelHeight} px · 1:1";
        RedrawOverlay();
        UpdateStatus();
    }

    /// <summary>Путь из CSV («data/source/...») разрешается относительно корня проекта.</summary>
    private string ResolveImagePath(string relative)
    {
        var normalized = relative.Replace('/', Path.DirectorySeparatorChar);
        return Path.IsPathRooted(normalized) ? normalized : Path.Combine(_root, normalized);
    }

    private BitmapSource LoadImage(string path)
    {
        if (_imageCache.TryGetValue(path, out var cached))
        {
            return cached;
        }

        var image = ImageLoader.Load(path);
        _imageCache[path] = image;
        _cacheOrder.Add(path);

        while (_cacheOrder.Count > ImageCacheLimit)
        {
            var oldest = _cacheOrder[0];
            _cacheOrder.RemoveAt(0);
            _imageCache.Remove(oldest);
        }

        return image;
    }

    private void ClearImage(string message)
    {
        _image = null;
        ImageView.Source = null;
        Placeholder.Text = message;
        Placeholder.Visibility = Visibility.Visible;
        Title = "boxes-explorer";
    }

    // ---------- отрисовка маски ----------

    private void RedrawOverlay()
    {
        Overlay.Children.Clear();

        if (_image is null)
        {
            return;
        }

        if (_points.Count >= 3)
        {
            var polygon = new Polygon
            {
                Stroke = SelectionStrokeBrush,
                StrokeThickness = 1,
                Fill = _points.Count == MaxPoints ? SelectionFillBrush : null,
                IsHitTestVisible = false,
            };

            foreach (var point in _points)
            {
                polygon.Points.Add(new Point(point.X + 0.5, point.Y + 0.5));
            }

            Overlay.Children.Add(polygon);
        }
        else if (_points.Count == 2)
        {
            Overlay.Children.Add(CreateLine(_points[0], _points[1], SelectionStrokeBrush, dashed: false));
        }

        // Пока область не замкнута — пунктир от последней точки к курсору
        // (и замыкающая сторона, когда точек уже три).
        if (_points.Count is > 0 and < MaxPoints && _hoverPoint is { } hover)
        {
            Overlay.Children.Add(CreateLine(_points[^1], hover, PreviewStrokeBrush, dashed: true));

            if (_points.Count == 3)
            {
                Overlay.Children.Add(CreateLine(hover, _points[0], PreviewStrokeBrush, dashed: true));
            }
        }

        for (var i = 0; i < _points.Count; i++)
        {
            AddMarker(_points[i], PointBrushes[i], i + 1);
        }
    }

    private static Line CreateLine((int X, int Y) from, (int X, int Y) to, Brush stroke, bool dashed) => new()
    {
        X1 = from.X + 0.5,
        Y1 = from.Y + 0.5,
        X2 = to.X + 0.5,
        Y2 = to.Y + 0.5,
        Stroke = stroke,
        StrokeThickness = 1,
        StrokeDashArray = dashed ? new DoubleCollection([4.0, 3.0]) : null,
        IsHitTestVisible = false,
    };

    private void AddMarker((int X, int Y) point, Brush brush, int number)
    {
        var cx = point.X + 0.5;
        var cy = point.Y + 0.5;

        Overlay.Children.Add(new Line
        {
            X1 = cx - MarkerArm,
            Y1 = cy,
            X2 = cx + MarkerArm,
            Y2 = cy,
            Stroke = brush,
            StrokeThickness = 1,
        });

        Overlay.Children.Add(new Line
        {
            X1 = cx,
            Y1 = cy - MarkerArm,
            X2 = cx,
            Y2 = cy + MarkerArm,
            Stroke = brush,
            StrokeThickness = 1,
        });

        var ring = new Ellipse
        {
            Width = 5,
            Height = 5,
            Stroke = brush,
            StrokeThickness = 1,
            IsHitTestVisible = false,
        };

        Canvas.SetLeft(ring, cx - 2.5);
        Canvas.SetTop(ring, cy - 2.5);
        Overlay.Children.Add(ring);

        var label = new TextBlock
        {
            Text = number.ToString(CultureInfo.InvariantCulture),
            Foreground = brush,
            FontSize = 11,
            FontWeight = FontWeights.SemiBold,
            IsHitTestVisible = false,
        };

        Canvas.SetLeft(label, cx + MarkerArm + 1);
        Canvas.SetTop(label, cy - MarkerArm - 12);
        Overlay.Children.Add(label);
    }

    // ---------- счётчики и статус ----------

    private void UpdateCounters()
    {
        if (_document is null)
        {
            FilledText.Text = "";
            return;
        }

        var boxes = _document.Rows.Count(row => row.HasBox);
        var kinds = _document.Rows.Count(row => row.Kind.Trim().Length > 0);
        FilledText.Text = $"с точками {boxes} из {_document.Rows.Count} · kind {kinds}";
    }

    private void RebuildOptions()
    {
        if (_document is null)
        {
            return;
        }

        var kinds = new List<string>(CanonicalKinds);
        foreach (var row in _document.Rows)
        {
            var kind = row.Kind.Trim();
            if (kind.Length > 0 && !kinds.Contains(kind, StringComparer.OrdinalIgnoreCase))
            {
                kinds.Add(kind);
            }
        }

        _kindOptions.Clear();
        foreach (var kind in kinds)
        {
            _kindOptions.Add(kind);
        }
    }

    private void UpdateStatus()
    {
        var text = _rowText.Length > 0 ? _rowText + "    |    " : "";
        text += $"Курсор: {_cursorText}";

        for (var i = 0; i < MaxPoints; i++)
        {
            text += $"    {i + 1}: " + (i < _points.Count ? FormatPoint(_points[i]) : "—");
        }

        if (!string.IsNullOrEmpty(_message))
        {
            text += $"    |    {_message}";
        }

        StatusText.Text = text;
    }

    private static string FormatPoint((int X, int Y) point) =>
        string.Create(CultureInfo.InvariantCulture, $"{point.X}, {point.Y}");

    private static Brush CreateBrush(Color color)
    {
        var brush = new SolidColorBrush(color);
        brush.Freeze();
        return brush;
    }
}
