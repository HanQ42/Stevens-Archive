//
//  WeatherRequestManager.swift
//  MyWeather2
//
//  Created by Hanqing Liu on 12/2/20.
//

import Foundation
import CoreLocation

protocol WeatherRequestManagerDelegate {
    func didUpdateWeather(_ weatherRequestManager: WeatherRequestManager, weather: WeatherModel)
    func didFailWithError(error: Error)
}

struct WeatherRequestManager {
    let weatherURL = "https://api.openweathermap.org/data/2.5/weather?appid=8d0a4a418c19ea5de1710ab0aacfbb55&units=imperial"
    
    var delegate: WeatherRequestManagerDelegate?
    
    func fetchWeather(latitude: CLLocationDegrees, longitude: CLLocationDegrees) {
        let urlString = "\(weatherURL)&lat=\(latitude)&lon=\(longitude)"
        performRequest(with: urlString)
    }
    
    func performRequest(with urlString: String) {
        if let url = URL(string: urlString) {
            let session = URLSession(configuration: .default)
            let task = session.dataTask(with: url) { (data, response, error) in
                DispatchQueue.main.async {
                    if error != nil {
                        delegate?.didFailWithError(error: error!)
                        return
                    } else {
                        if let safeData = data {
                            if let weather = self.parseJson(safeData) {
                                self.delegate?.didUpdateWeather(self, weather: weather)
                            } else {
                                self.delegate?.didFailWithError(error: error!)
                            }
                        }
                    }
                }
            }
            task.resume()
        }
    }
    func parseJson(_ weatherData: Data) -> WeatherModel? {
        let decoder = JSONDecoder()
        do {
            let decodedData = try decoder.decode(WeatherData.self, from: weatherData)
            
            let id = decodedData.weather[0].id
            let description = decodedData.weather[0].description
            let temp = decodedData.main.temp
            let pressure = decodedData.main.pressure
            let humidity = decodedData.main.humidity
            let windSpeed = decodedData.wind.speed
            let windDegree = decodedData.wind.deg
            let cloudiness = decodedData.clouds.all
            let visibility = decodedData.visibility
            let name = decodedData.name
            
            let weather = WeatherModel(conditionId: id, cityName: name, description: description, temperature: temp, windSpeed: windSpeed, WindDegree: windDegree, pressure: pressure, humidity: humidity, cloudiness: cloudiness, visibility: visibility)
            return weather
        } catch {
            delegate?.didFailWithError(error: error)
            return nil
        }
    }
}
