using CoinVision.Api;
using CoinVision.Api.Controllers;
using CoinVision.Api.Services;
using Microsoft.AspNetCore.Http.Features;
using Scalar.AspNetCore;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

builder.Host.UseSerilog((context, configuration) => configuration
    .ReadFrom.Configuration(context.Configuration)
    .Enrich.FromLogContext()
    .WriteTo.Console(
        outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}"));

builder.Services.Configure<ModelOptions>(builder.Configuration.GetSection(ModelOptions.Section));
builder.Services.Configure<UploadOptions>(builder.Configuration.GetSection(UploadOptions.Section));

builder.Services.AddSingleton<OnnxModelRunner>();
builder.Services.AddSingleton<IImagePreprocessor, ImagePreprocessor>();
builder.Services.AddScoped<PredictionService>();
builder.Services.AddHostedService<ModelWarmupService>();

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddOpenApi("v1", options => options.AddDocumentTransformer((document, _, _) =>
{
    document.Info = new()
    {
        Title = "CoinVision API",
        Version = "v1",
        Description = "Year recognition for coins. Upload one or more images, get the year, the confidence and the timings.",
    };
    return Task.CompletedTask;
}));
builder.Services.AddProblemDetails();
builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 512L * 1024 * 1024;
    options.ValueCountLimit = 4096;
    options.MultipartHeadersLengthLimit = 64 * 1024;
});
builder.Services.AddCors(options => options.AddDefaultPolicy(policy => policy
    .WithOrigins("http://localhost:5173", "http://127.0.0.1:5173")
    .AllowAnyHeader()
    .AllowAnyMethod()));

var app = builder.Build();

app.UseExceptionHandler();
app.UseStatusCodePages();
app.UseSerilogRequestLogging(options =>
{
    options.MessageTemplate = "HTTP {RequestMethod} {RequestPath} -> {StatusCode} in {Elapsed:0.0} ms";
    options.GetLevel = (context, _, exception) =>
        exception is not null || context.Response.StatusCode >= 500
            ? Serilog.Events.LogEventLevel.Error
            : Serilog.Events.LogEventLevel.Information;
});

app.MapOpenApi();
app.MapScalarApiReference(options => options
    .WithTitle("CoinVision API")
    .WithDefaultHttpClient(ScalarTarget.CSharp, ScalarClient.HttpClient));

app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapControllers();
app.MapFallbackToFile("index.html");

Log.Information("CoinVision API starting, model directory {Directory}", app.Configuration["Model:Directory"]);
app.Run();
