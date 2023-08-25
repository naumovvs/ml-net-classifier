using Microsoft.ML;

using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;
using SKKClassifier;
using System.Diagnostics;

List<DataRecord> ReadFromXLSX(string fname)
{
    Console.WriteLine(fname);
    List<DataRecord> records = new List<DataRecord>();

    using (SpreadsheetDocument doc = SpreadsheetDocument.Open(fname, false))
    {
        Sheet? sheet = doc.WorkbookPart?.Workbook.Sheets?.GetFirstChild<Sheet>();
        Worksheet? worksheet = (doc.WorkbookPart?.GetPartById(sheet.Id.Value) as WorksheetPart)?.Worksheet;

        IEnumerable<Row>? rows = worksheet?.GetFirstChild<SheetData>()?.Elements<Row>();
        Console.WriteLine($"#rows={rows?.Count()}");

        for (int i = 0; i < rows?.Count(); i++)
        {
            Row? row = rows.ElementAt(i);
            IEnumerable<Cell>? cells = row.Elements<Cell>();
            //Console.WriteLine($"row {row?.RowIndex?.Value}: #cells={cells.Count()}");

            string text = GetCellValue(doc, cells.ElementAt(2));
            //Console.WriteLine(text);

            if (text == "ModuleCode")
            {
                string target = GetCellValue(doc, cells.ElementAt(0));
                string sensorId = GetCellValue(doc, cells.ElementAt(1));
                //Console.WriteLine($"{sensorId}: {target}");

                List<float> tSeq = new List<float>();
                List<float> lSeq = new List<float>();
                List<DateTime> dSeq = new List<DateTime>();

                for (int j = 3; j < cells.Count(); j++)
                {
                    string date = GetCellValue(doc, cells.ElementAt(j));
                    if (date != "")
                    {
                        dSeq.Add(DateTime.FromOADate(float.Parse(date)));
                    }
                }

                IEnumerable<Cell>? lightCells = rows.ElementAt(i + 1).Elements<Cell>();
                IEnumerable<Cell>? tempCells = rows.ElementAt(i + 2).Elements<Cell>();
                string tempTitle = GetCellValue(doc, tempCells.ElementAt(0));
                string lightTitle = GetCellValue(doc, lightCells.ElementAt(0));
                
                if (lightTitle == "ISL29035_Light")
                {
                    for (int j = 1; j < lightCells.Count(); j++)
                    {
                        string light = GetCellValue(doc, lightCells.ElementAt(j));
                        lSeq.Add(float.Parse(light));
                    }
                }

                if (tempTitle == "SHT3X_Temperature")
                {
                    for (int j = 1; j < tempCells.Count(); j++)
                    {
                        string temp = GetCellValue(doc, tempCells.ElementAt(j));
                        tSeq.Add(float.Parse(temp));
                    }
                }

                //Console.WriteLine($"target={target}, id={sensorId}, #tSeq={tSeq.Count}, #lSeq={lSeq.Count}");
                if (tSeq.Count() > 0 && lSeq.Count() > 0) // tSeq.Count() == lSeq.Count()
                {
                    DataRecord record = new(id: sensorId, tempSeq: tSeq,
                        lightSeq: lSeq, dateSeq: dSeq) { Target = target };
                    records.Add(record);
                }   
            }
        }    
    }

    return records;
}

string GetCellValue(SpreadsheetDocument doc, Cell cell)
{
    string? value = cell.InnerText;
    if (cell.DataType != null && cell.DataType.Value == CellValues.SharedString)
    {
        value = cell.CellValue?.InnerText;
        return doc.WorkbookPart.SharedStringTablePart.SharedStringTable
               .ChildElements.GetItem(int.Parse(value)).InnerText;
    }
    return value;
}

List<DataRecord> PreprocessData(List<DataRecord> data, int length = 50)
{
    List<DataRecord> processed = new List<DataRecord>();

    if (data.Count() == 0)
    {
        return processed;
    }

    int minSize = data[0].TempArray.Length;
    int maxSize = data[0].TempArray.Length;
    double avg = 0;

    foreach (DataRecord d in data)
    {
        int size = d.TempArray.Length;
        if (size < minSize) minSize = size;
        if (size > maxSize) maxSize = size;
        avg += size;
    }
    avg /= data.Count();

    Debug.WriteLine($"minSize={minSize}, maxSize={maxSize}, avg={avg}");

    foreach (DataRecord d in data)
    {
        int size = d.TempArray.Length;
        if (size >= length)
        {
            List<float> seq = new List<float>();
            for (int i = 0; i < length; i++)
            {
                seq.Add(d.TempArray[i]);
            }

            DataRecord dr = new DataRecord(id: d.SensorId, tempSeq: seq,
                lightSeq: null, dateSeq: null) { Target = d.Target };

            processed.Add(dr);
        }
    }

    return processed;
}


List<DataRecord> drs = ReadFromXLSX("dataset_all_vs.xlsx"); // rawData
//Console.WriteLine($"#records read = {rawData.Count}");
//List<DataRecord> drs = PreprocessData(rawData, length: 1000);
Console.WriteLine($"#records selected = {drs.Count}");
//foreach (DataRecord dr in drs)
//{
//    Console.WriteLine($"sensor={dr.SensorId}, target={dr.Target}, #seq={dr.SeqArray.Count()}");
//    foreach (var ar in dr.SeqArray)
//    {
//        Console.Write(ar + " ");
//    }
//    Console.WriteLine();
//}

MLContext mlContext = new MLContext();

// 1. Import training data
IDataView data = mlContext.Data.LoadFromEnumerable(drs);

// 2. Specify data preparation and model training pipeline
DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Target")
    .Append(mlContext.MulticlassClassification.Trainers
        //.SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features"))
        //.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
        .LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
        //.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

// 3. Train the model 
var model = pipeline.Fit(trainData);
Console.WriteLine("Training finished.");

// 4. Estimate model quality
var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testData));
Console.WriteLine($"microaccuracy={metrics.MicroAccuracy}");
Console.WriteLine($"macroaccuracy={metrics.MacroAccuracy}");
//var engine = mlContext.Model.CreatePredictionEngine<DataRecord, Prediction>(model);
//Console.WriteLine($"pred={engine.Predict(drs[0]).Target}, pred={engine.Predict(drs[1]).Target}");

Console.ReadKey();