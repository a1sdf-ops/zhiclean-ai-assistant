// Package weather provides weather data via 和风天气 (QWeather) API.
package weather

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// cityToID maps Chinese city names to QWeather location IDs.
var cityToID = map[string]string{
	"北京": "101010100", "上海": "101020100", "天津": "101030100", "重庆": "101040100",
	"广州": "101280101", "深圳": "101280601", "东莞": "101281601", "佛山": "101280301",
	"珠海": "101280701", "惠州": "101280301", "中山": "101281701", "汕头": "101280501",
	"杭州": "101210101", "宁波": "101210401", "温州": "101210701",
	"南京": "101190101", "苏州": "101190401", "无锡": "101190201",
	"成都": "101270101", "武汉": "101200101", "长沙": "101250101",
	"合肥": "101220101", "芜湖": "101220301",
	"福州": "101230101", "厦门": "101230201", "泉州": "101230501",
	"济南": "101120101", "青岛": "101120201", "烟台": "101120501",
	"郑州": "101180101", "洛阳": "101180901",
	"石家庄": "101090101", "唐山": "101090501", "保定": "101090201",
	"西安": "101110101", "咸阳": "101110200",
	"沈阳": "101070101", "大连": "101070201",
	"哈尔滨": "101050101", "长春": "101060101", "太原": "101100101",
	"呼和浩特": "101080101", "南昌": "101240101", "南宁": "101300101",
	"海口": "101310101", "贵阳": "101260101", "昆明": "101290101",
	"拉萨": "101140101", "兰州": "101160101", "西宁": "101150101",
	"银川": "101170101", "乌鲁木齐": "101130101",
	"香港": "101320101", "澳门": "101330101", "台北": "101340101",
}

// NowData represents the current weather response from QWeather API.
type NowData struct {
	Now struct {
		Temp     string `json:"temp"`
		Text     string `json:"text"`
		Humidity string `json:"humidity"`
		WindDir  string `json:"windDir"`
		WindScale string `json:"windScale"`
	} `json:"now"`
}

// GetWeather returns formatted weather string for the given city.
func GetWeather(city string) string {
	apiKey := os.Getenv("QWEATHER_API_KEY")
	cityID, ok := cityToID[city]
	if !ok {
		return fmt.Sprintf("未找到城市「%s」的天气信息", city)
	}

	// If no API key, return mock data for demo
	if apiKey == "" {
		return mockWeather(city)
	}

	url := fmt.Sprintf("https://devapi.qweather.com/v7/weather/now?location=%s&key=%s", cityID, apiKey)

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Sprintf("无法获取城市「%s」的天气数据: %v", city, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Sprintf("读取天气数据失败: %v", err)
	}

	// Parse raw to handle nested structure
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return fmt.Sprintf("解析天气数据失败: %v", err)
	}

	if code, ok := raw["code"]; ok {
		var codeStr string
		json.Unmarshal(code, &codeStr)
		if codeStr != "200" {
			return fmt.Sprintf("天气API返回错误(code=%s)", codeStr)
		}
	}

	var nowRaw map[string]json.RawMessage
	if nowBytes, ok := raw["now"]; ok {
		json.Unmarshal(nowBytes, &nowRaw)
	}

	getStr := func(key string) string {
		if v, ok := nowRaw[key]; ok {
			var s string
			json.Unmarshal(v, &s)
			return s
		}
		return "N/A"
	}

	return fmt.Sprintf("【%s实时天气】%s，%s°C，湿度%s%%，%s%s级",
		city, getStr("text"), getStr("temp"),
		getStr("humidity"), getStr("windDir"), getStr("windScale"))
}

// mockWeather returns demo weather data when no API key configured.
func mockWeather(city string) string {
	return fmt.Sprintf("【%s实时天气（演示数据）】晴间多云，26°C，湿度55%%，东南风3级", city)
}
