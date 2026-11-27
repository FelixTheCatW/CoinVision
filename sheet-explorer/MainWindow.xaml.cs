using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace SheetExplorer;

/// <summary>
/// Проверка verdict.csv: слева исходное изображение выбранной строки в масштабе 1:1,
/// справа таблица; значения колонок kind и note правятся и сразу сохраняются в файл.
/// </summary>
public partial class MainWindow : Window
{
    private const string CsvRelativePath = @"data\annotations\verdict\verdict.csv";
    private const int ImageCacheLimit = 4;

    private static readonly string[] CanonicalKinds = ["engraved", "slab"];

    private readonly ObservableCollection<string> _kindOptions = [];
    private readonly ObservableCollection<string> _noteOptions = [];
    private readonly Dictionary<string, BitmapSource> _imageCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _cacheOrder = [];
    private readonly DispatcherTimer _saveTimer;

    private VerdictDocument? _document;
    private string _root = "";
    private bool _saving;
    private bool _dirty;
    private bool _editing;

    public MainWindow()
    {
        InitializeComponent();

        _saveTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(400) };
        _saveTimer.Tick += OnSaveTimerTick;

        Loaded += OnLoaded;
    }

    /// <summary>Подсказки для выпадающего списка колонки kind.</summary>
    public ObservableCollection<string> KindOptions => _kindOptions;

    /// <summary>Подсказки для выпадающего списка колонки note (уже введённые значения).</summary>
    public ObservableCollection<string> NoteOptions => _noteOptions;

    // ---------- загрузка файла ----------

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            LoadDocument();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "sheet-explorer", MessageBoxButton.OK, MessageBoxImage.Error);
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
        var document = VerdictDocument.Load(csvPath);

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
            Table.SelectedIndex = 0;
            Table.ScrollIntoView(Table.SelectedItem);
            Table.Focus();
        }
        else
        {
            ClearImage("Файл пуст.");
        }
    }

    /// <summary>Ищет корень проекта: каталог, в котором лежит data\annotations\verdict\verdict.csv.</summary>
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

    // ---------- сохранение ----------

    private void OnRowChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (_saving)
        {
            return;
        }

        _dirty = true;
        SaveStateText.Text = "Не сохранено…";
        _saveTimer.Stop();
        _saveTimer.Start();
        UpdateCounters();
    }

    private void OnSaveTimerTick(object? sender, EventArgs e)
    {
        _saveTimer.Stop();
        Save(commitEdit: false);
    }

    private void OnSaveClick(object sender, RoutedEventArgs e) => Save(commitEdit: true);

    private void OnPreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.S && Keyboard.Modifiers == ModifierKeys.Control)
        {
            e.Handled = true;
            Save(commitEdit: true);
            return;
        }

        // 1 / 2 — проставить первый/второй вариант kind и перейти на строку ниже.
        if (Keyboard.Modifiers == ModifierKeys.None && !_editing && !IsTextInputFocused())
        {
            var index = e.Key switch
            {
                Key.D1 or Key.NumPad1 => 0,
                Key.D2 or Key.NumPad2 => 1,
                _ => -1,
            };

            if (index >= 0)
            {
                e.Handled = true;
                ApplyKindAndAdvance(index);
            }
        }
    }

    /// <summary>Клавиши 1/2: подставить вариант из списка kind и перейти на следующую строку.</summary>
    private void ApplyKindAndAdvance(int index)
    {
        if (index >= _kindOptions.Count)
        {
            return;
        }

        if (Table.SelectedItem is not VerdictRow row)
        {
            SaveStateText.Text = "Сначала выделите строку";
            return;
        }

        row.Kind = _kindOptions[index];

        if (Table.SelectedIndex < Table.Items.Count - 1)
        {
            Table.SelectedIndex++;
            Table.ScrollIntoView(Table.SelectedItem);
        }

        Save(commitEdit: false);
    }

    private static bool IsTextInputFocused() =>
        Keyboard.FocusedElement is TextBoxBase or ComboBox;

    private void OnWindowClosing(object? sender, CancelEventArgs e)
    {
        if (_document is null)
        {
            return;
        }

        if (!Save(commitEdit: true))
        {
            e.Cancel = true;
        }
    }

    /// <summary>
    /// Записывает файл, если были изменения. <paramref name="commitEdit"/> завершает
    /// редактирование текущей ячейки, чтобы введённый текст точно попал в файл.
    /// </summary>
    private bool Save(bool commitEdit)
    {
        if (_document is null)
        {
            return true;
        }

        if (commitEdit)
        {
            CommitPendingEdit();
        }

        _saveTimer.Stop();
        if (!_dirty)
        {
            return true;
        }

        _saving = true;
        try
        {
            _document.Save();
            _dirty = false;
            SaveStateText.Text = $"Сохранено в {DateTime.Now:HH:mm:ss}";
            return true;
        }
        catch (Exception ex)
        {
            SaveStateText.Text = "Ошибка сохранения";
            MessageBox.Show(
                this,
                ex.Message + "\n\nФайл: " + _document.Path,
                "Не удалось сохранить verdict.csv",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return false;
        }
        finally
        {
            _saving = false;
            if (!_editing)
            {
                RebuildOptions();
            }
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

    private void UpdateCounters()
    {
        if (_document is null)
        {
            FilledText.Text = "";
            return;
        }

        var filled = _document.Rows.Count(row => row.Kind.Trim().Length > 0 || row.Note.Trim().Length > 0);
        FilledText.Text = $"заполнено {filled} из {_document.Rows.Count}";
    }

    // ---------- выпадающие списки ----------

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

        ReplaceItems(_kindOptions, kinds);

        var notes = _document.Rows
            .Select(row => row.Note.Trim())
            .Where(note => note.Length > 0)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(note => note, StringComparer.OrdinalIgnoreCase)
            .ToList();
        ReplaceItems(_noteOptions, notes);
    }

    private static void ReplaceItems(ObservableCollection<string> target, List<string> values)
    {
        target.Clear();
        foreach (var value in values)
        {
            target.Add(value);
        }
    }

    private void OnBeginningEdit(object? sender, DataGridBeginningEditEventArgs e) => _editing = true;

    private void OnCellEditEnding(object? sender, DataGridCellEditEndingEventArgs e)
    {
        _editing = false;
        Dispatcher.BeginInvoke(DispatcherPriority.Background, new Action(RebuildOptions));
    }

    // ---------- изображение ----------

    private void OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        ShowSelectedImage();

        // Догрузка после того, как DataGrid зафиксирует правку предыдущей строки.
        _saveTimer.Stop();
        Dispatcher.BeginInvoke(DispatcherPriority.Background, new Action(() =>
        {
            if (_document is not null)
            {
                Save(commitEdit: false);
            }
        }));
    }

    private void ShowSelectedImage()
    {
        if (_document is null)
        {
            return;
        }

        if (Table.SelectedItem is not VerdictRow row)
        {
            ClearImage("Выберите строку в таблице —\nизображение из первой колонки откроется здесь в масштабе 1:1");
            StatusText.Text = "";
            return;
        }

        var relative = row.Image.Trim();
        var prefix = $"Строка {Table.SelectedIndex + 1} из {_document.Rows.Count} · ";

        if (relative.Length == 0)
        {
            ClearImage("В первой колонке этой строки нет пути к изображению.");
            StatusText.Text = prefix + "путь не задан";
            return;
        }

        var full = ResolveImagePath(relative);
        if (!File.Exists(full))
        {
            ClearImage("Файл не найден:\n" + relative);
            StatusText.Text = prefix + relative + " · файл не найден";
            return;
        }

        try
        {
            var image = LoadImage(full);
            ImageView.Source = image;
            Placeholder.Visibility = Visibility.Collapsed;
            Scroll.ScrollToHome();
            StatusText.Text = $"{prefix}{relative} · {image.PixelWidth}×{image.PixelHeight} px · 1:1";
            Title = $"{Path.GetFileName(full)} — sheet-explorer";
        }
        catch (Exception ex)
        {
            ClearImage("Не удалось открыть изображение:\n" + ex.Message);
            StatusText.Text = prefix + relative;
        }
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
        ImageView.Source = null;
        Placeholder.Text = message;
        Placeholder.Visibility = Visibility.Visible;
        Title = "sheet-explorer";
    }
}
