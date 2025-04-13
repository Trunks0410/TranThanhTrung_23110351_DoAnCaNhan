# 🔢 Đồ án cá nhân: 8-Puzzle Solver

## 🎯 Mục tiêu
Xây dựng một chương trình giải bài toán **8-Puzzle** sử dụng nhiều thuật toán tìm kiếm khác nhau trong lĩnh vực Trí tuệ nhân tạo.

---

## 🧠 Các thuật toán được triển khai

| Thuật Toán               | Mô Tả                                                                 | Minh Hóa GIF                              |
|--------------------------|----------------------------------------------------------------------|-------------------------------------------|
| **Breadth-First Search (BFS)** | Khám phá tuần tự theo từng lớp: Bắt đầu từ nút gốc, thuật toán duyệt qua tất cả các nút ở cùng độ sâu trước khi chuyển sang độ sâu tiếp theo. Ưu điểm: Đảm bảo tìm được đường đi ngắn nhất (nếu có). Nhược điểm: Tốn nhiều bộ nhớ cho các bài toán có không gian trạng thái rộng.             | <img src="images/bfs.gif" width="500" alt="BFS"> |
| **Depth-First Search (DFS)**   | Tìm kiếm theo chiều sâu, có thể không tìm được đường ngắn nhất.    | <img src="images/dfs.gif" width="500" alt="DFS"> |
| **Uniform Cost Search (UCS)**  | Tìm kiếm dựa trên chi phí, tương tự BFS nhưng với trọng số.        | <img src="images/ucs.gif" width="500" alt="UCS"> |
| **Iterative Deepening DFS (IDDFS)** | Kết hợp DFS và giới hạn độ sâu, hiệu quả hơn DFS.                 | <img src="images/iddfs.gif" width="500" alt="IDDFS"> |
| **Greedy Best-First Search**   | Sử dụng heuristic để ưu tiên trạng thái hứa hẹn nhất.             | <img src="images/greedy.gif" width="500" alt="GREEDY"> |
| **A* Search**                 | Kết hợp chi phí và heuristic, tìm đường ngắn nhất hiệu quả.        | <img src="images/astar.gif" width="500" alt="A*"> |
| **IDA* Search**               | Phiên bản tối ưu của A* với giới hạn chi phí.                     | <img src="images/idastar.gif" width="500" alt="IDA*"> |
| **Simple Hill Climbing**       | Leo dốc đơn giản, dễ kẹt ở cực trị cục bộ.                       | <img src="images/simplehillclimbing.gif" width="500" alt="Simple HC"> |
| **Steepest Hill Climbing**     | Kiểm tra tất cả lân cận, chọn tốt nhất, nhưng vẫn có thể kẹt.     | <img src="images/steepesthillclimbing.gif" width="500" alt="Steepest HC"> |
| **Stochastic Hill Climbing**   | Leo dốc ngẫu nhiên, tránh cực trị cục bộ tốt hơn.                | <img src="images/stochastichillclimbing.gif" width="500" alt="Stochastic HC"> |
| **Simulated Annealing**        | Sử dụng nhiệt độ để chấp nhận giải pháp xấu, tìm giải toàn cục.    | <img src="images/simulatedannealing.gif" width="500" alt="Simulated Annealing"> |
| **Beam Search**                | Tìm kiếm chùm, giữ lại một số lượng cố định trạng thái tốt nhất.   | <img src="images/beamsearch.gif" width="500" alt="Beam Search"> |

## 👨‍💻 Tác giả

**Trần Thành Trung**  
MSSV: `23110351`  
Môn: `Trí Tuệ Nhân Tạo`
Giáo viên hướng dẫn: `Phan Thị Huyền Trang` 

---
